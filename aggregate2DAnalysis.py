#!/usr/bin/env python
'''
Aggregate peak analaysis for a single chromosome

Usage: aggregatePeakAnalysis.py -i <bedpe> -r <res> [options] <cmap>

Options:
    -i <bedpe>      input regions to aggregate in bedpe format
    -m <mode>       "center": align regions on the center; "interval": align whole interval [default: center]
    -o <outprfx>    output prefix [default: ]
    -r <res>        resolution in bp
    -f <nflnk>      number of flanking bins [default: 10]
    -s <nscale>     number of internal bins to scale the whole interval to, needed by "-m <mode>" [default: 10]
    -d <minD>       minimum distance, default set to ceiling(sqrt(2)*nflnk)*res
    -z <mnzp>       minimum proportion of submatrix value being non-zero [default: 0.0]
    --rmbg          remove background from distance decay
    --full          write full submatrices values
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
from os.path import dirname,isdir
from os import makedirs
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from scipy.sparse import load_npz
from scipy.interpolate import RectBivariateSpline


def readRegions(filename):
    regions = pd.read_table(filename, header=None,
            names=['c1','s1','e1','c2','s2','e2'],
            dtype={'c1':str,'s1':int,'e1':int,'c2':str,'s2':int,'e2':int})
    return regions


def getSubmatrixIdxForLoops(regions, res, n, nflnk, minD=None):
    """
    res: resolution in bp
    n: cMap dimension in bins
    nflnk: number of flanking bins
    minD: minimum distance between loop anchor in bp
    """
    regions['m1'] = ((regions.s1+regions.e1)/2).astype(int)
    regions['m2'] = ((regions.s2+regions.e2)/2).astype(int)
    if minD is None:
        minD = np.ceil(nflnk*np.sqrt(2)) * res
    regions = regions[np.abs(regions.m1-regions.m2)>=minD]

    regions['idx1'] = np.ceil((regions.m1-res+0.1)/res).astype(int)
    regions['idx2'] = np.ceil((regions.m2-res+0.1)/res).astype(int)
    regions['idx_x1'] = regions.idx1 - nflnk
    regions['idx_x2'] = regions.idx1 + nflnk + 1
    regions['idx_y1'] = regions.idx2 - nflnk
    regions['idx_y2'] = regions.idx2 + nflnk + 1
    return regions


def getSubmatrixIdxForCompartments(regions, res, n, nflnk):
    """
    res: resolution in bp
    n: cMap dimension in bins
    nflnk: number of flanking bins
    """
    regions['idx_s1'] = np.ceil((regions.s1-res+0.1)/res).astype(int)
    regions['idx_e1'] = np.ceil((regions.e1-res+0.1)/res).astype(int)
    regions['idx_s2'] = np.ceil((regions.s2-res+0.1)/res).astype(int)
    regions['idx_e2'] = np.ceil((regions.e2-res+0.1)/res).astype(int)
    regions['idx_x1'] = regions.idx_s1 - nflnk
    regions['idx_x2'] = regions.idx_e1 + nflnk + 1
    regions['idx_y1'] = regions.idx_s2 - nflnk
    regions['idx_y2'] = regions.idx_e2 + nflnk + 1
    return regions


def getSubmatrixIdx(regions, mode, res, n, nflnk, minD=None):
    if mode == 'center':
        regions = getSubmatrixIdxForLoops(regions, res, n, nflnk, minD)
    else:
        regions = getSubmatrixIdxForCompartments(regions, res, n, nflnk)
    regions = regions[(regions.idx_x1>=0) & (regions.idx_x2<=n)
            & (regions.idx_y1>=0) & (regions.idx_y2<=n)]
    return regions


def makeDistanceMatrix(n):
    d = np.arange(n, dtype=int)
    dmat = np.repeat(d.reshape(1,n), n, axis=0) - d.reshape(n,1)
    dmat[dmat<0] = np.abs(dmat[dmat<0])
    return dmat


def extractSubMatrix(matrix, regions, bgmat=None):
    if bgmat is not None:
        submats = [matrix[i1:i2,j1:j2] / bgmat[i1:i2,j1:j2]
                for i1,i2,j1,j2 in list(zip(
                    regions.idx_x1, regions.idx_x2,
                    regions.idx_y1, regions.idx_y2))]
    else:
        submats = [matrix[i1:i2,j1:j2] for i1,i2,j1,j2 in list(zip(
            regions.idx_x1, regions.idx_x2, regions.idx_y1, regions.idx_y2))]
    for i,sm in enumerate(submats):
        submats[i][np.isnan(sm)] = np.nanmedian(sm)
        submats[i][np.isinf(sm)] = np.nanmedian(sm)
    return submats


def scaleCompartments(mat, nflnk, nscale):
    M,N = mat.shape
    logging.debug(mat.shape)
    m1 = nflnk
    m2 = M - nflnk
    n1 = m1
    n2 = N - nflnk
    m = m2 - m1
    n = n2 - n1
    m1x = np.arange(m1)
    n1x = m1x
    mx = np.arange(m)
    nx = np.arange(n)
    mxs = np.linspace(0,m-1,nscale)
    nxs = np.linspace(0,n-1,nscale)

    top = mat[0:m1,n1:n2]
    logging.debug(top.shape)
    left = mat[m1:m2,0:n1]
    logging.debug(left.shape)
    bottom = mat[m2:M,n1:n2]
    right = mat[m1:m2,n2:N]
    center = mat[m1:m2,n1:n2]
    logging.debug(center.shape)
    topleft = mat[0:m1,0:n1]
    topright = mat[0:m1,n2:N]
    bottomleft = mat[m2:M,0:n1]
    bottomright = mat[m2:M,n2:N]

    kf = min(3, nflnk-1)
    km = min(3, m-1)
    kn = min(3, n-1)
    top_rbs = RectBivariateSpline(m1x, nx, top, kx=kf, ky=kn)
    left_rbs = RectBivariateSpline(mx, n1x, left, kx=km, ky=kf)
    bottom_rbs = RectBivariateSpline(m1x, nx, bottom, kx=kf, ky=kn)
    right_rbs = RectBivariateSpline(mx, n1x, right, kx=km, ky=kf)
    center_rbs = RectBivariateSpline(mx, nx, center, kx=km, ky=kn)

    top_scaled = top_rbs(m1x,nxs)
    left_scaled = left_rbs(mxs,n1x)
    bottom_scaled = bottom_rbs(m1x,nxs)
    right_scaled = right_rbs(mxs,n1x)
    center_scaled = center_rbs(mxs,nxs)

    Ns = nscale + 2*nflnk
    n2s = nflnk + nscale
    new_mat = np.zeros((Ns, Ns))
    logging.debug(new_mat.shape)
    new_mat[0:n1,0:n1] = topleft
    new_mat[0:n1,n1:n2s] = top_scaled
    new_mat[0:n1,n2s:Ns] = topright
    new_mat[n1:n2s,0:n1] = left_scaled
    new_mat[n1:n2s,n1:n2s] = center_scaled
    new_mat[n1:n2s,n2s:Ns] = right_scaled
    new_mat[n2s:Ns,0:n1] = bottomleft
    new_mat[n2s:Ns,n1:n2s] = bottom_scaled
    new_mat[n2s:Ns,n2s:Ns] = bottomright

    return new_mat


def matrixSparsity(matrix):
    return np.nansum(matrix>0).astype(float) / np.prod(matrix.shape)


def maskCenter(nflnk):
    n = 2*nflnk + 1
    mask = np.zeros((n,n), dtype=np.bool)
    mask[nflnk,nflnk] = True
    return mask


def maskBottomLeft(nflnk):
    n = 2*nflnk + 1
    m = int(nflnk/2)
    mask = np.zeros((n,n), dtype=np.bool)
    mask[(n-m):n,0:m] = True
    return mask


def calcApaStats(mats, nflnk):
    n = mats[0].shape[0]
    n1 = nflnk
    n2 = n - nflnk
    m = n2 - n1
    sums = np.nansum(np.nansum(mats, axis=1), axis=1)
    center_sums = np.nansum(np.nansum(mats[:,n1:n2,n1:n2], axis=1), axis=1)
    top_sums = np.nansum(np.nansum(mats[:,0:n1,n1:n2], axis=1), axis=1)
    left_sums = np.nansum(np.nansum(mats[:,n1:n2,0:n1], axis=1), axis=1)
    bottom_sums = np.nansum(np.nansum(mats[:,n2:n,n1:n2], axis=1), axis=1)
    right_sums = np.nansum(np.nansum(mats[:,n1:n2,n2:n], axis=1), axis=1)
    topleft_sums = np.nansum(np.nansum(mats[:,0:n1,0:n1], axis=1), axis=1)
    topright_sums = np.nansum(np.nansum(mats[:,0:n1,n2:n], axis=1), axis=1)
    bottomleft_sums = np.nansum(np.nansum(mats[:,n2:n,0:n1], axis=1), axis=1)
    bottomright_sums = np.nansum(np.nansum(mats[:,n2:n,n2:n], axis=1), axis=1)
    center_means = center_sums / m / m
    bg_means = (sums - center_sums) / (n*n - m*m)
    top_means = top_sums / n1 / m
    left_means = left_sums / m / n1
    bottom_means = bottom_sums / n1 / m
    right_means = right_sums / m / n1
    topleft_means = topleft_sums / n1 / n1
    topright_means = topright_sums / n1 / n1
    bottomleft_means = bottomleft_sums / n1 / n1
    bottomright_means = bottomright_sums / n1 / n1
    return (center_means,bg_means,top_means,left_means,bottom_means,right_means,
            topleft_means,bottomleft_means,bottomright_means,topright_means)


def calcDistanceDecay(matrix, dmat):
    n = matrix.shape[0]
    k = np.tri(n, dtype=bool)
    d = np.arange(n, dtype=int)
    x0 = dmat[~k]
    y0 = matrix[~k]
    dd = np.bincount(x0, weights=y0) / np.arange(n, 0, -1, dtype=int)
    return dd[dmat]


def main(args):
    logging.info(args)
    regionFn = args['i']
    outprfx = args['o']
    mode = args['m']
    res = int(args['r'])
    minD = args['d']
    if minD is not None:
        minD = int(minD)
    nflnk = int(args['f'])
    nscale = int(args['s'])
    mnzp = float(args['z'])
    cmapFn = args['cmap']
    rmbg = args['rmbg']
    full = args['full']

    regions = readRegions(regionFn)
    assert len(regions) > 0, 'No regions'

    matrix = load_npz(cmapFn).toarray()
    matrix[np.isnan(matrix)] = 0
    n = matrix.shape[0]
    if rmbg:
        dmat = makeDistanceMatrix(n)
        bgmat = calcDistanceDecay(matrix, dmat)
    else:
        bgmat = None

    regions = getSubmatrixIdx(regions, mode, res, n, nflnk, minD)

    submats = extractSubMatrix(matrix, regions, bgmat=bgmat)
    if mode == 'center':
        width = 2*nflnk+1
        k_correct_shape = np.array([sm.shape==(width,width) for sm in submats])
    else:
        k_correct_shape = np.array([len(sm.shape)==2 and min(sm.shape)>2*nflnk+1
            for sm in submats])
    submat_sparsity = np.array([matrixSparsity(sm) for sm in submats])
    k_include = np.logical_and(k_correct_shape, submat_sparsity > mnzp)
    assert sum(k_include) > 0, 'No submats'

    submats = [submats[i] for i in np.where(k_include)[0]]

    if mode == 'center':
        submats = np.array(submats)
    else:
        submats = np.array([scaleCompartments(sm, nflnk, nscale)
            for sm in submats])
    matMean = np.nanmean(submats, axis=0)
    matMean[np.isnan(matMean)] = np.nanmedian(matMean)
    matMean[np.isinf(matMean)] = np.nanmedian(matMean)

    try:
        makedirs(dirname(outprfx))
    except OSError as e:
        if e.errno == errno.EEXIST and isdir(dirname(outprfx)):
            pass
        else:
            raise

    if rmbg:
        typ = 'rmbgAPA'
    else:
        typ = 'APA'
    np.savetxt(outprfx+typ+'.txt', matMean, delimiter='\t', fmt='%.2e')
    if full:
        np.save(outprfx+typ+'.submats', submats)

    center,bg,T,L,B,R,TL,BL,BR,TR = calcApaStats(submats, nflnk)
    stats = pd.DataFrame({
        'chrom'    :regions.c1[k_include],
        'start'    :regions.s1[k_include],
        'end'      :regions.s2[k_include],
        'center'   :center,
        'bg'       :bg,
        'T'        :T,
        'L'        :L,
        'B'        :B,
        'R'        :R,
        'TL'       :TL,
        'BL'       :BL,
        'BR'       :BR,
        'TR'       :TR})
    stats.to_csv(outprfx+typ+'.stat', sep='\t', float_format='%.2e',
            columns=['chrom','start','end','center','bg','T','L','B','R','TL','BL','BR','TR'],
            header=False, index=False)
    return 0


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
