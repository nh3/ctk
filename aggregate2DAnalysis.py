#!/usr/bin/env python
'''
Aggregate peak analaysis for a single chromosome

Usage: aggregate2DAnalysis.py -i <bedpe> -r <res> [options] <cmap>

Options:
    -i <bedpe>      input regions to aggregate in DOUBLE bedpe format (12 columns)
    -o <outprfx>    output prefix [default: ]
    -r <res>        resolution in bp
    -f <nflnk>      number of flanking bins [default: 5]
    -c <ncntr>      number of center bins to scale the whole interval to [default: 10]
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
            names=['c1','s1','e1','c2','s2','e2','c1c','s1c','e1c','c2c','s2c','e2c'],
            dtype={'c1':str,'s1':int,'e1':int,'c2':str,'s2':int,'e2':int,
                'c1c':str,'s1c':int,'e1c':int,'c2c':str,'s2c':int,'e2c':int})
    return regions


def getSubmatrixIdx(regions, res, n):
    """
    res: resolution in bp
    n: cMap dimension in bins
    """
    regions['idx_x1'] = np.ceil((regions.s1-res+0.1)/res).astype(int)
    regions['idx_x2'] = np.ceil((regions.e1-res+0.1)/res).astype(int)
    regions['idx_y1'] = np.ceil((regions.s2-res+0.1)/res).astype(int)
    regions['idx_y2'] = np.ceil((regions.e2-res+0.1)/res).astype(int)
    regions['idx_x1c'] = np.ceil((regions.s1c-res+0.1)/res).astype(int)
    regions['idx_x2c'] = np.ceil((regions.e1c-res+0.1)/res).astype(int)
    regions['idx_y1c'] = np.ceil((regions.s2c-res+0.1)/res).astype(int)
    regions['idx_y2c'] = np.ceil((regions.e2c-res+0.1)/res).astype(int)
    regions = regions[(regions.idx_x1>=0) & (regions.idx_x2<=n)
            & (regions.idx_y1>=0) & (regions.idx_y2<=n)]
    regions = regions[(regions.idx_x1<regions.idx_x1c-1) & (regions.idx_x2c<regions.idx_x2-1) &
            (regions.idx_x1c<regions.idx_x2c-1) & (regions.idx_y1c<regions.idx_y2c-1) &
            (regions.idx_y1<regions.idx_y1c-1) & (regions.idx_y2c<regions.idx_y2-1)]
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


def scaleCompartments(mat, region, nflnk, ncntr):
    logging.debug(mat.shape)
    Ns = ncntr + 2*nflnk
    ns = nflnk + ncntr
    new_mat = np.zeros((Ns, Ns))

    X,Y = mat.shape
    x1 = int(region.idx_x1c - region.idx_x1)
    x2 = int(region.idx_x2c - region.idx_x1)
    y1 = int(region.idx_y1c - region.idx_y1)
    y2 = int(region.idx_y2c - region.idx_y1)
    m1,m2,m3 = x1,x2-x1,X-x2
    n1,n2,n3 = y1,y2-y1,Y-y2

    m1x = np.arange(m1)
    n1x = np.arange(n1)
    m2x = np.arange(m2)
    n2x = np.arange(n2)
    m3x = np.arange(m3)
    n3x = np.arange(n3)
    m1s = np.linspace(0,m1-1,nflnk)
    n1s = np.linspace(0,n1-1,nflnk)
    m2s = np.linspace(0,m2-1,ncntr)
    n2s = np.linspace(0,n2-1,ncntr)
    m3s = np.linspace(0,m3-1,nflnk)
    n3s = np.linspace(0,n3-1,nflnk)

    top = mat[0:x1,y1:y2]
    logging.debug(top.shape)
    left = mat[x1:x2,0:y1]
    logging.debug(left.shape)
    bottom = mat[x2:X,y1:y2]
    right = mat[x1:x2,y2:Y]
    center = mat[x1:x2,y1:y2]
    logging.debug(center.shape)
    topleft = mat[0:x1,0:y1]
    topright = mat[0:x1,y2:Y]
    bottomleft = mat[x2:X,0:y1]
    bottomright = mat[x2:X,y2:Y]

    km1 = min(3, m1-1)
    kn1 = min(3, n1-1)
    km2 = min(3, m2-1)
    kn2 = min(3, n2-1)
    km3 = min(3, m3-1)
    kn3 = min(3, n3-1)
    top_rbs = RectBivariateSpline(m1x, n2x, top, kx=km1, ky=kn2)
    left_rbs = RectBivariateSpline(m2x, n1x, left, kx=km2, ky=kn1)
    bottom_rbs = RectBivariateSpline(m3x, n2x, bottom, kx=km3, ky=kn2)
    right_rbs = RectBivariateSpline(m2x, n3x, right, kx=km2, ky=kn3)
    center_rbs = RectBivariateSpline(m2x, n2x, center, kx=km2, ky=kn2)
    topleft_rbs = RectBivariateSpline(m1x, n1x, topleft, kx=km1, ky=kn1)
    bottomleft_rbs = RectBivariateSpline(m3x, n1x, bottomleft, kx=km3, ky=kn1)
    topright_rbs = RectBivariateSpline(m1x, n3x, topright, kx=km1, ky=kn3)
    bottomright_rbs = RectBivariateSpline(m3x, n3x, bottomright, kx=km3, ky=kn3)

    top_scaled = top_rbs(m1s,n2s)
    left_scaled = left_rbs(m2s,n1s)
    bottom_scaled = bottom_rbs(m3s,n2s)
    right_scaled = right_rbs(m2s,n3s)
    center_scaled = center_rbs(m2s,n2s)
    topleft_scaled = topleft_rbs(m1s,n1s)
    bottomleft_scaled = bottomleft_rbs(m3s,n1s)
    bottomright_scaled = bottomright_rbs(m3s,n3s)
    topright_scaled = topright_rbs(m1s,n3s)

    new_mat[0:nflnk,0:nflnk] = topleft_scaled
    new_mat[0:nflnk,nflnk:ns] = top_scaled
    new_mat[0:nflnk,ns:Ns] = topright_scaled
    new_mat[nflnk:ns,0:nflnk] = left_scaled
    new_mat[nflnk:ns,nflnk:ns] = center_scaled
    new_mat[nflnk:ns,ns:Ns] = right_scaled
    new_mat[ns:Ns,0:nflnk] = bottomleft_scaled
    new_mat[ns:Ns,nflnk:ns] = bottom_scaled
    new_mat[ns:Ns,ns:Ns] = bottomright_scaled

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
    res = int(args['r'])
    nflnk = int(args['f'])
    ncntr = int(args['c'])
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

    regions = getSubmatrixIdx(regions, res, n)
    assert regions.shape[0]>0, 'No submats'

    submats = extractSubMatrix(matrix, regions, bgmat=bgmat)

    submat_sparsity = np.array([matrixSparsity(sm) for sm in submats])
    k_include = submat_sparsity > mnzp
    assert sum(k_include) > 0, 'No submats'

    submats = [submats[i] for i in np.where(k_include)[0]]
    regions = regions[k_include]

    submats = np.array([scaleCompartments(submats[i], regions.iloc[[i]], nflnk, ncntr)
        for i in xrange(len(submats))])
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
        'chrom'    :regions.c1,
        'start'    :regions.s1c,
        'end'      :regions.s2c,
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
