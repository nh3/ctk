#!/usr/bin/env python
'''
Calculate insulation score from C-map

Usage: cMapInsulation.py -r <res> -o <prfx> [options] <input>

Options:
    -r <res>    resolution in bp
    -o <prfx>   output prefix
    -s <size>   size of insulation block in bp [default: 100000]
    -d <minD>   minimum distance from the diagonal in bp [default: 0]
    -w <width>  width for smoothing by running mean in number of bins [default: 11]
    -z <minS>   minimum insulation domain size in number of bins [default: 5]
    -v <minV>   minimum depth from both sides of the local minimums [default: 0.5]
    <input>     input sparse matrix
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy.sparse
import pandas as pd
from scipy.interpolate import splrep,splev


def moving_average(x, n=5):
    mask = np.isnan(x)
    k = np.ones(n, dtype=int)
    sums = np.convolve(np.where(mask,0,x), k, mode='same')
    counts = np.convolve(~mask,k,mode='same')
    return sums/counts


def smooth_by_MA(x, w, t=1):
    sm_x = x
    for i in range(t):
        sm_x = moving_average(sm_x, w)
    return sm_x


def robust_scale(x, center=True):
    m = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x-m))*1.482602
    if center:
        y = (x-m)/mad
    else:
        y = x/mad
    return y


def rlencode(x, dropNA=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropNA: bool, optional
        Drop all runs of NaNs.
    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))
    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    if dropNA:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    return starts, lengths, values


def calculate_insulation(mat, z, d):
    N = mat.shape[0]
    k = np.nansum(mat, axis=1) == 0
    mat[:,k] = np.nan
    mat[k,:] = np.nan
    x = np.arange(N)
    I1 = np.maximum(x-d-z, 0)
    I2 = np.maximum(x-d, 0)
    J1 = np.minimum(x+d, N)
    J2 = np.minimum(x+d+z, N)
    insBlocks = [mat[i1:i2,j1:j2] for i1,i2,j1,j2 in zip(I1, I2, J1, J2)]
    insBlockSums = np.array([np.nansum(blk) for blk in insBlocks])
    insBlockSizes = np.array(map(np.product, map(np.shape, insBlocks)))
    insBlockNanCounts = np.array(map(np.sum, map(np.isnan, insBlocks)))
    insBlockMeans = insBlockSums / (insBlockSizes - insBlockNanCounts)
    return insBlockMeans,insBlockSizes-insBlockNanCounts


def prune_small_rle(rle, minL):
    indices = rle[0]
    lengths = rle[1]
    values = rle[2]
    N = indices.shape[0]
    i_left = np.where(lengths<=minL)[0]
    i_right = i_left + 1
    i = np.union1d(i_left, i_right)
    i = i[i<N]
    k = np.setdiff1d(np.arange(N), i)
    return indices[k],lengths[k],values[k]


def find_local_minimums(x, dx, ddx, minL):
    N = x.shape[0]
    dx_rle = prune_small_rle(rlencode(dx>0), minL)
    ddx_rle = prune_small_rle(rlencode(ddx>0), minL)

    dx0_idx = dx_rle[0]
    LMIN_idx = dx0_idx[ddx[dx0_idx]>0]
    LMAX_idx = dx0_idx[ddx[dx0_idx]<0]
    neighbor_max_dx0_idx = []
    for idx in LMIN_idx:
        left_max_indices = LMAX_idx[LMAX_idx<idx]
        if left_max_indices.shape[0] > 0:
            left_max_idx = left_max_indices[-1]
        else:
            left_max_idx = 0
        right_max_indices = LMAX_idx[LMAX_idx>idx]
        if right_max_indices.shape[0] > 0:
            right_max_idx = right_max_indices[0]
        else:
            right_max_idx = N-1
        neighbor_max_dx0_idx.append([left_max_idx,right_max_idx,idx])
    x_depth = np.array([(x[i]-x[k],x[j]-x[k]) for i,j,k in neighbor_max_dx0_idx])

    ddx0_idx = ddx_rle[0]
    neighbor_ddx0_idx = []
    for idx in LMIN_idx:
        left_ddx0_indices = ddx0_idx[ddx0_idx<idx]
        if left_ddx0_indices.shape[0] > 0:
            left_ddx0_idx = left_ddx0_indices[-1]
        else:
            left_ddx0_idx = 0
        right_ddx0_indices = ddx0_idx[ddx0_idx>idx]
        if right_ddx0_indices.shape[0] > 0:
            right_ddx0_idx = right_ddx0_indices[0]
        else:
            right_ddx0_idx = N-1
        neighbor_ddx0_idx.append([left_ddx0_idx,right_ddx0_idx,idx])
    dx_depth = np.array([(dx[i]-dx[k],dx[j]-dx[k]) for i,j,k in neighbor_ddx0_idx])
    n = LMIN_idx.shape[0]
    return pd.DataFrame(np.hstack([LMIN_idx.reshape(n,1),x_depth,dx_depth]),
            columns=['idx','depth_left','depth_right','slope_left','slope_right'])


def main(args):
    logging.info(args)
    mat = scipy.sparse.load_npz(args['input']).astype(float).toarray()
    outPrfx = args['o']
    res = int(args['r'])
    insZ = int(args['s'])
    insD = int(args['d'])
    min_s = int(args['z'])
    min_v = float(args['v'])
    w = int(args['w'])
    res_boost = 5

    N = mat.shape[0]
    d = insD/res
    z = insZ/res

    ins,insN = calculate_insulation(mat, z, d)
    smoothed_ins = smooth_by_MA(ins, w, 3)

    x = np.arange(N)*res
    k = np.isnan(smoothed_ins)
    f = splrep(x[~k], smoothed_ins[~k], k=5)
    xnew = np.arange(0, N*res, res/res_boost)
    xnew = xnew[xnew<=np.nanmax(x)]
    y = robust_scale(splev(xnew, f, der=0, ext=3))
    dy = robust_scale(splev(xnew, f, der=1, ext=3), center=False)
    ddy = robust_scale(splev(xnew, f, der=2, ext=3), center=False)

    smoothed_y = smooth_by_MA(y, w, 5)
    smoothed_dy = smooth_by_MA(dy, w, 5)
    smoothed_ddy = smooth_by_MA(ddy, w, 10)

    insulations = pd.DataFrame({'x':xnew, 'y':y, 'dy':dy, 'ddy':ddy, 'ys':smoothed_y, 'dys':smoothed_dy, 'ddys':smoothed_ddy})
    boundaries = find_local_minimums(smoothed_y, smoothed_dy, smoothed_ddy, min_s*res_boost)
    boundaries['x'] = boundaries.idx*int(res/res_boost) + int(res/res_boost/2)
    boundaries['called'] = (boundaries.depth_left>min_v) & (boundaries.depth_right>min_v)

    insFn = outPrfx + '.insulation.txt'
    bdFn = outPrfx + '.boundary.txt'

    insulations[['x','y','ys','dys','ddys']].to_csv(insFn, sep='\t', na_rep='NA', header=True, index=False)
    boundaries.to_csv(bdFn, sep='\t', na_rep='NA', header=True, index=False)


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
