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
from numpy.lib.recfunctions import append_fields
from scipy.sparse import load_npz
from scipy.stats import ttest_ind, ttest_1samp
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages                                                                                                                                           
from matplotlib.colors import LinearSegmentedColormap


def readLoops(filename):
    loops = np.loadtxt(filename, delimiter='\t',
            dtype={'names':('c1','s1','e1','c2','s2','e2'),
                   'formats':((np.unicode_,8), int, int,
                              (np.unicode_,8), int, int)})
    m1 = (loops['s1']+loops['e1'])/2
    m2 = (loops['s2']+loops['e2'])/2
    return append_fields(loops, ('m1','m2'), (m1,m2), dtypes=(int,int))


def filterLoops(loops, res, nflnk, n, minD=None):
    if minD is None:
        minD = np.ceil(nflnk*np.sqrt(2)) * res
    minDB = minD / res
    k = np.logical_and(np.abs(loops['idx1']-loops['idx2'])>=minDB,
            np.logical_and(loops['idx1']>nflnk, loops['idx1']<n-nflnk),
            np.logical_and(loops['idx2']>nflnk, loops['idx2']<n-nflnk))
    return loops[k]


def mapLoops(loops, res):
    idx1 = np.ceil((loops['m1']-res+0.1)/res)
    idx2 = np.ceil((loops['m2']-res+0.1)/res)
    return append_fields(loops, ('idx1','idx2'), (idx1,idx2), dtypes=(int,int))


def makeDistanceMatrix(n):
    d = np.arange(n, dtype=int)
    dmat = np.repeat(d.reshape(1,n), n, axis=0) - d.reshape(n,1)
    dmat[dmat<0] = np.abs(dmat[dmat<0])
    return dmat


def extractSubMatrix(matrix, dmat, loops, nflnk, bgmat=None):
    if bgmat is not None:
        submats = [matrix[(i-nflnk):(i+nflnk+1),(j-nflnk):(j+nflnk+1)]
                / bgmat[(i-nflnk):(i+nflnk+1),(j-nflnk):(j+nflnk+1)]
                for i,j in list(zip(loops['idx1'], loops['idx2']))]
    else:
        submats = [matrix[(i-nflnk):(i+nflnk+1),(j-nflnk):(j+nflnk+1)]
                for i,j in list(zip(loops['idx1'], loops['idx2']))]
    return submats


def filterMatrix(matrix, nonZeroProp=0.1):
    x = np.sum(matrix>0).astype(float) / np.prod(matrix.shape)
    return x > nonZeroProp


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


def calcApaStatsV1(matrix):
    n = matrix.shape[0]
    nflnk = (n-1)/2
    k_mid = maskMid(nflnk)
    mid = matrix[k_mid][0]
    rest = matrix[~k_mid]
    fc_rest = mid/np.mean(rest)
    t_rest,p_rest = ttest_1samp(matrix[~k_mid], mid)
    k_LL = maskLowerLeft(nflnk)
    LL = matrix[k_LL]
    fc_LL = mid/np.mean(LL)
    t_LL,p_LL = ttest_1samp(matrix[k_LL], mid)
    return fc_rest,p_rest,fc_LL,p_LL


def calcApaStatsV2(mats):
    n = mats[0].shape[0]
    nflnk = (n-1)/2
    k_mid = maskMid(nflnk)
    k_LL = maskLowerLeft(nflnk)
    mids = np.array([m[k_mid][0] for m in mats])
    rest_means = np.array([np.nanmean(m[~k_mid]) for m in mats])
    LL_means = np.array([np.nanmean(m[k_LL]) for m in mats])

    fc_rest = np.nanmean(mids)/np.nanmean(rest_means)
    t_rest,p_rest = ttest_ind(mids, rest_means, equal_var=False, nan_policy='omit')
    fc_LL = np.nanmean(mids)/np.nanmean(LL_means)
    t_LL,p_LL = ttest_ind(mids, LL_means, equal_var=False, nan_policy='omit')
    return fc_rest,p_rest,fc_LL,p_LL


def calcApaStatsV3(mats):
    n = mats[0].shape[0]
    nflnk = (n-1)/2
    k_mid = maskMid(nflnk)
    k_LL = maskLowerLeft(nflnk)
    mids = np.array([m[k_mid][0] for m in mats])
    rest_means = np.array([np.nanmean(m[~k_mid]) for m in mats])
    LL_means = np.array([np.nanmean(m[k_LL]) for m in mats])

    #fc_rest = np.nanmean(mids)/np.nanmean(rest_means)
    #t_rest,p_rest = ttest_ind(mids, rest_means, equal_var=False, nan_policy='omit')
    #fc_LL = np.nanmean(mids)/np.nanmean(LL_means)
    #t_LL,p_LL = ttest_ind(mids, LL_means, equal_var=False, nan_policy='omit')
    return np.array([mids,rest_means,LL_means]).transpose()


def calcDistanceDecay(matrix, dmat):
    n = matrix.shape[0]
    k = np.tri(n, dtype=bool)
    d = np.arange(n, dtype=int)
    x0 = dmat[~k]
    y0 = matrix[~k]
    dd = np.bincount(x0, weights=y0) / np.arange(n, 0, -1, dtype=int)
    return dd[dmat]


def plotHeatmap(matMean, outpdf):
    RdWh = LinearSegmentedColormap.from_list('RdWh', [(0,'white'),(1,'red')])
    plt.imshow(matMean, cmap=RdWh)
    plt.axis('off')
    pp = PdfPages(outpdf)                                                                                                                                                                      
    plt.savefig(pp, format='pdf')
    pp.close()

def saveAPA(submats, outprfx, rmbg=False):
    matMean = np.mean(np.array(submats), axis=0)
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
    #plotHeatmap(matMean, outprfx + typ + '.pdf')

    #fc_rest,p_rest,fc_ll,p_ll = calcApaStatsV1(matmean)
    #fc_rest,p_rest,fc_ll,p_ll = calcApaStatsV2(submats)
    stats = calcApaStatsV3(submats)
    np.savetxt(outprfx+typ+'.stat', stats, delimiter='\t', fmt='%.2e')
    #with open(outprfx + typ + '.stat', 'w') as fh:
    #    print('{}\t{}\t{}\t{}'.format(outprfx, len(submats), fc_rest, p_rest),file=fh)


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

    submats = extractSubMatrix(matrix, dmat, loops, nflnk, bgmat=bgmat)
    submats = [sm for sm in submats if filterMatrix(sm, nonZeroProp=mnzp)]
    submats = [sm for sm in submats if sm.shape == (width,width)]
    assert len(submats) > 0, 'No submats'

    saveAPA(submats, outprfx, rmbg)


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
