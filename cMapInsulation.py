#!/usr/bin/env python
'''
Usage: cMapInsulation.py [options] <input>

Options:
    -r <res>    resolution in bp
    -d <minD>   minimum distance from the diagonal in bp [default: 0]
    -s <size>   size of insulation block in bp [default: 100000]
    -w <width>  width for smoothing by running mean in bp [default: 5000]
    <input>     input sparse matrix
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import numpy as np
import scipy.sparse
import pandas as pd

"""
Eight-order approximation centered at grid point derived from:
    Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
        Mathematics of Computation 51 (184): 699-706, doi:10.1090/S0025-5718-1988-0935077-0
"""
def prepare_derivative_kernel(n):
    kernels = np.array([
        [-1.0/280, -4.0/105,  1.0/5, -4.0/5,         0, 4.0/5, -1.0/5, 4.0/105, -1.0/280],
        [-1.0/560,  8.0/315, -1.0/5,  8.0/5, -205.0/72, 8.0/5, -1.0/5, 8.0/315, -1.0/560]])
    return kernels[n-1]


def prepare_smooth_kernel(width, repeats=1):
    rolling_mean_kernel = np.ones(width)/float(width)
    kernel = rolling_mean_kernel
    for i in range(repeats):
        kernel = np.convolve(kernel, rolling_mean_kernel, mode='same')
    return kernel


#def prepare_derivative_kernel(n, width, smooth_times=1):
#    kernels = np.array([
#        [-1.0/280, -4.0/105,  1.0/5, -4.0/5,         0, 4.0/5, -1.0/5, 4.0/105, -1.0/280],
#        [-1.0/560,  8.0/315, -1.0/5,  8.0/5, -205.0/72, 8.0/5, -1.0/5, 8.0/315, -1.0/560]])
#    rolling_mean_kernel = np.ones(width)/float(width)
#    for i in range(smooth_times):
#        kernel = np.convolve(kernels[n-1], rolling_mean_kernel)
#    return 1e6*kernel[::-1]


def find_concave_regions(y, kernel, chrom='.', tol=1e-10):
    d2y = np.convolve(y, kernel, mode='same')                                                                                                                                   
    s = np.where(np.diff((d2y < tol).astype(int))==1)[0] + 1
    e = np.where(np.diff((d2y < tol).astype(int))==-1)[0] + 1
    if s[0] > e[0]:
        s = np.insert(s, 0, 0)
    if s[-1] > e[-1]:
        e = np.insert(e, len(e), len(y))
    v = [-np.mean(d2y[i:j]) for i,j in zip(s,e)]
    regions = np.array(list(zip([chrom]*len(s),s,e,v)), dtype=[('chrom',np.unicode_,8),('start',int),('end',int),('value',float)])
    return regions


def main(args):
    logging.info(args)
    mat = scipy.sparse.load_npz(args['input']).astype(float)
    res = int(args['r'])
    minD = int(args['d'])
    insZ = int(args['s'])
    smthW = int(args['w'])
    d = minD/res
    z = insZ/res
    w = smthW/res
    N = mat.shape[0]
    x = np.arange(N)
    I1 = np.maximum(x-d-z, 0)
    I2 = np.maximum(x-d, 0)
    J1 = np.minimum(x+d, N)
    J2 = np.minimum(x+d+z, N)
    insBlocks = [mat[i1:i2,j1:j2] for i1,i2,j1,j2 in zip(I1, I2, J1, J2)]
    insBlockSums = np.array([np.nansum(blk.toarray()) for blk in insBlocks])
    insBlockSizes = np.array(map(np.product, map(np.shape, insBlocks)))
    insBlockMeans = insBlockSums / insBlockSizes
    d1kernel = prepare_derivative_kernel(1)
    d2kernel = prepare_derivative_kernel(2)
    smooth_kernel = prepare_smooth_kernel(w, 3)
    #d1kernel = prepare_derivative_kernel(1, w, 3)
    #d2kernel = prepare_derivative_kernel(2, w, 3)
    #concave_regions = find_concave_regions(insBlockMeans, d2kernel)
    #np.savetxt(sys.stdout, concave_regions, fmt=['%s','%d','%d','%.4f'], delimiter='\t')
    insBlockSmoothMeans = np.convolve(insBlockMeans, smooth_kernel, mode='same')

    D1 = np.convolve(insBlockMeans, d1kernel, mode='same')
    D1s = np.convolve(insBlockSmoothMeans, d1kernel, mode='same')
    D2 = np.convolve(insBlockMeans, d2kernel, mode='same')
    D2s = np.convolve(insBlockSmoothMeans, d2kernel, mode='same')
    insTable = pd.DataFrame({'size':insBlockSizes, 'sum':insBlockSums, 'mean':insBlockMeans, 'x':x, 'D1':D1, 'D2':D2, 'D1s':D1s, 'D2s':D2s})
    insTable.to_csv(sys.stdout, sep='\t', na_rep='NA', header=True, index=False)


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
