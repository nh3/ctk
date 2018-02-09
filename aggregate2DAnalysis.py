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
from scipy.interpolate import UnivariateSpline,RectBivariateSpline


def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


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
    regions['idx_x1'] = np.round(regions.s1/res).astype(int)
    regions['idx_x2'] = np.round(regions.e1/res).astype(int)
    regions['idx_y1'] = np.round(regions.s2/res).astype(int)
    regions['idx_y2'] = np.round(regions.e2/res).astype(int)
    regions['idx_x1c'] = np.round(regions.s1c/res).astype(int)
    regions['idx_x2c'] = np.round(regions.e1c/res).astype(int)
    regions['idx_y1c'] = np.round(regions.s2c/res).astype(int)
    regions['idx_y2c'] = np.round(regions.e2c/res).astype(int)
    regions = regions[(regions.idx_x1>=0) & (regions.idx_x2<=n)
            & (regions.idx_y1>=0) & (regions.idx_y2<=n)]
    regions = regions[(regions.idx_x1<regions.idx_x1c) & (regions.idx_x2c<regions.idx_x2) &
            (regions.idx_x1c<regions.idx_x2c) & (regions.idx_y1c<regions.idx_y2c) &
            (regions.idx_y1<regions.idx_y1c) & (regions.idx_y2c<regions.idx_y2)]
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


@memodict
def scaleIdx(shape):
    n,ns = shape
    nx = np.arange(n)
    nsx = np.linspace(0,n-1,ns)
    return nx,nsx


def scaleMatrix(mat, ms, ns):
    m,n = mat.shape
    if m == 1 and n == 1:
        tgt = np.full((ms,ns), mat[0,0])
    elif m == 1 and n > 1:
        nx,nsx = scaleIdx((n,ns))
        kn = min(3, n-1)
        spl = UnivariateSpline(nx,mat[0], k=kn)
        tgt = np.repeat(spl(nsx).reshape((1,ns)), ms, axis=0)
    elif m > 1 and n == 1:
        mx,msx = scaleIdx((m,ms))
        km = min(3, m-1)
        spl = UnivariateSpline(mx,mat[:,0], k=km)
        tgt = np.repeat(spl(msx).reshape((ms,1)), ns, axis=1)
    else:
        mx,msx = scaleIdx((m,ms))
        nx,nsx = scaleIdx((n,ns))
        km = min(3, m-1)
        kn = min(3, n-1)
        rbs = RectBivariateSpline(mx,nx,mat,kx=km,ky=kn)
        tgt = rbs(msx,nsx)
    return tgt


def scaleCompartments(mat, region, nflnk, ncntr):
    logging.debug(mat.shape)
    X,Y = mat.shape
    Ns = ncntr + 2*nflnk

    if X==Ns and Y==Ns:
        return mat

    ns = nflnk + ncntr
    new_mat = np.zeros((Ns, Ns))

    x1 = int(region.idx_x1c - region.idx_x1)
    x2 = int(region.idx_x2c - region.idx_x1)
    y1 = int(region.idx_y1c - region.idx_y1)
    y2 = int(region.idx_y2c - region.idx_y1)

    top = mat[0:x1,y1:y2]
    left = mat[x1:x2,0:y1]
    bottom = mat[x2:X,y1:y2]
    right = mat[x1:x2,y2:Y]
    center = mat[x1:x2,y1:y2]
    topleft = mat[0:x1,0:y1]
    topright = mat[0:x1,y2:Y]
    bottomleft = mat[x2:X,0:y1]
    bottomright = mat[x2:X,y2:Y]

    new_mat[0:nflnk,0:nflnk] = scaleMatrix(topleft, nflnk, nflnk)
    new_mat[0:nflnk,nflnk:ns] = scaleMatrix(top, nflnk, ncntr)
    new_mat[0:nflnk,ns:Ns] = scaleMatrix(topright, nflnk, nflnk)
    new_mat[nflnk:ns,0:nflnk] = scaleMatrix(left, ncntr, nflnk)
    new_mat[nflnk:ns,nflnk:ns] = scaleMatrix(center, ncntr, ncntr)
    new_mat[nflnk:ns,ns:Ns] = scaleMatrix(right, ncntr, nflnk)
    new_mat[ns:Ns,0:nflnk] = scaleMatrix(bottomleft, nflnk, nflnk)
    new_mat[ns:Ns,nflnk:ns] = scaleMatrix(bottom, nflnk, ncntr)
    new_mat[ns:Ns,ns:Ns] = scaleMatrix(bottomright, nflnk, nflnk)
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
