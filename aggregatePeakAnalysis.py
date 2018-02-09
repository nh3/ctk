#!/usr/bin/env python
'''
Aggregate peak analaysis for a single chromosome

Usage: aggregatePeakAnalysis.py -i <loops> -r <res> [options] <cmap>

Options:
    -i <loops>      input loops in bedpe format
    -o <outprfx>    output prefix [default: ]
    -r <res>        resolution in bp
    -f <nflnk>      number of flanking bins [default: 10]
    -d <minD>       minimum distance, default set to ceiling(sqrt(2)*nflnk)*res
    -z <mnzp>       minimum proportion of submatrix value being non-zero [default: 0.0]
    --rmbg          remove background from distance decay
    --prof          print profiling information
    --debug         print debug information
    <cmap>          input matrix in npz format
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import errno
import os.path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from scipy.sparse import load_npz


def readLoops(filename):
    loops = pd.read_table(filename, header=None,
            names=['c1','s1','e1','c2','s2','e2'],
            dtype={'c1':str,'s1':int,'e1':int,'c2':str,'s2':int,'e2':int})
    loops['m1'] = ((loops.s1+loops.e1)/2).astype(int)
    loops['m2'] = ((loops.s2+loops.e2)/2).astype(int)
    return loops


def filterLoops(loops, res, nflnk, n, minD=None):
    if minD is None:
        minD = np.ceil(nflnk*np.sqrt(2)) * res
    minDB = minD / res
    return loops[(np.abs(loops.idx1-loops.idx2)>=minDB)
            & (loops.idx1>nflnk) & (loops.idx1<n-nflnk)
            & (loops.idx2>nflnk) & (loops.idx2<n-nflnk)]


def mapLoops(loops, res):
    loops['idx1'] = np.ceil((loops.m1-res+0.1)/res).astype(int)
    loops['idx2'] = np.ceil((loops.m2-res+0.1)/res).astype(int)
    return loops


def makeDistanceMatrix(n):
    d = np.arange(n, dtype=int)
    dmat = np.repeat(d.reshape(1,n), n, axis=0) - d.reshape(n,1)
    dmat[dmat<0] = np.abs(dmat[dmat<0])
    return dmat


def extractSubMatrix(matrix, loops, nflnk, bgmat=None):
    if bgmat is not None:
        submats = [matrix[(i-nflnk):(i+nflnk+1),(j-nflnk):(j+nflnk+1)]
                / bgmat[(i-nflnk):(i+nflnk+1),(j-nflnk):(j+nflnk+1)]
                for i,j in list(zip(loops.idx1, loops.idx2))]
    else:
        submats = [matrix[(i-nflnk):(i+nflnk+1),(j-nflnk):(j+nflnk+1)]
                for i,j in list(zip(loops.idx1, loops.idx2))]
    return submats


def filterMatrix(matrix, nonZeroProp=0.1):
    x = np.sum(matrix>0).astype(float) / np.prod(matrix.shape)
    return x > nonZeroProp


def matrixSparsity(matrix):
    return np.sum(matrix>0).astype(float) / np.prod(matrix.shape)


def maskMid(nflnk):
    n = 2*nflnk + 1
    mask = np.zeros((n,n), dtype=np.bool)
    mask[nflnk,nflnk] = True
    return mask


def maskLowerLeft(nflnk):
    n = 2*nflnk + 1
    m = int(nflnk/2)
    mask = np.zeros((n,n), dtype=np.bool)
    mask[(n-m):n,0:m] = True
    return mask


def calcApaStats(mats):
    n = mats[0].shape[0]
    nflnk = (n-1)/2
    k_mid = maskMid(nflnk)
    k_LL = maskLowerLeft(nflnk)
    mids = np.array([m[k_mid][0] for m in mats])
    rest_means = np.array([np.nanmean(m[~k_mid]) for m in mats])
    LL_means = np.array([np.nanmean(m[k_LL]) for m in mats])
    return mids,rest_means,LL_means


def calcDistanceDecay(matrix, dmat):
    n = matrix.shape[0]
    k = np.tri(n, dtype=bool)
    d = np.arange(n, dtype=int)
    x0 = dmat[~k]
    y0 = matrix[~k]
    dd = np.bincount(x0, weights=y0) / np.arange(n, 0, -1, dtype=int)
    return dd[dmat]


#def plotHeatmap(matMean, outpdf):
#    RdWh = LinearSegmentedColormap.from_list('RdWh', [(0,'white'),(1,'red')])
#    plt.imshow(matMean, cmap=RdWh)
#    plt.axis('off')
#    pp = PdfPages(outpdf)
#    plt.savefig(pp, format='pdf')
#    pp.close()


def main(args):
    logging.info(args)
    loopFn = args['i']
    outprfx = args['o']
    res = int(args['r'])
    minD = args['d']
    if minD is not None:
        minD = int(minD)
    nflnk = int(args['f'])
    mnzp = float(args['z'])
    cmapFn = args['cmap']
    rmbg = args['rmbg']
    width = 2*nflnk+1

    loops = readLoops(loopFn)
    assert len(loops) > 0, 'No loops'
    loops = mapLoops(loops, res)

    matrix = load_npz(cmapFn).toarray()
    matrix[np.isnan(matrix)] = 0
    n = matrix.shape[0]
    if rmbg:
        dmat = makeDistanceMatrix(n)
        bgmat = calcDistanceDecay(matrix, dmat)
    else:
        dmat = None
        bgmat = None
    loops = filterLoops(loops, res, nflnk, n, minD)

    submats = extractSubMatrix(matrix, loops, nflnk, bgmat=bgmat)
    k_correct_shape = np.array([sm.shape==(width,width) for sm in submats])
    submat_sparsity = np.array([matrixSparsity(sm) for sm in submats])
    k_include = np.logical_and(k_correct_shape, submat_sparsity > mnzp)
    assert sum(k_include) > 0, 'No submats'

    submats = [submats[i] for i in np.where(k_include)[0]]
    submats = np.array(submats)

    matMean = np.mean(submats, axis=0)
    matMean[np.isnan(matMean)] = np.nanmedian(matMean)

    try:
        os.makedirs(os.path.dirname(outprfx))
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(os.path.dirname(outprfx)):
            pass
        else:
            raise

    if rmbg:
        typ = 'rmbgAPA'
    else:
        typ = 'APA'
    np.savetxt(outprfx+typ+'.txt', matMean, delimiter='\t', fmt='%.2e')

    mids,rest_means,LL_means = calcApaStats(submats)
    stats = pd.DataFrame({
        'chrom'    :loops.c1[k_include],
        'start'    :loops.s1[k_include],
        'end'      :loops.s2[k_include],
        'mid'      :mids,
        'rest_mean':rest_means,
        'LL_mean'  :LL_means})
    stats.to_csv(outprfx+typ+'.stat', sep='\t', float_format='%.2e',
            columns=['chrom','start','end','mid','rest_mean','LL_mean'],
            header=False, index=False)


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.lstrip('-<').rstrip('>'):args[k] for k in args}
    try:
        if args.get('debug'):
            logLevel = logging.DEBUG
        else:
            logLevel = logging.WARN
        logging.basicConfig(
                level=logLevel,
                format='%(asctime)s; %(levelname)s; %(funcName)s; %(message)s',
                datefmt='%y-%m-%d %H:%M:%S')
        if args.get('prof'):
            import cProfile
            cProfile.run('main(args)')
        else:
            main(args)
    except KeyboardInterrupt:
        logging.warning('Interrupted')
        sys.exit(1)
