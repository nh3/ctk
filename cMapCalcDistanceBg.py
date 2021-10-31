#!/usr/bin/env python
'''
Calculate distance-dependent contact frequency background

Usage: cMapCalcDistanceBg.py [options] <cmap> <ddbg>

Options:
    --prof          print profiling information
    --debug         print debug information
    -r <res>        resolution [default: 1000]
    -f              force monotonic non-increasing
    -F              force monotonic decreasing, override -f
    <cmap>          input matrix in npz format
    <ddbg>          output dd background as one column table
'''

from __future__ import print_function
import sys
import signal
import logging
from os.path import dirname,isdir
from os import makedirs
import numpy as np
import scipy.sparse
from scipy.interpolate import LSQUnivariateSpline
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
np.seterr(divide='ignore', invalid='ignore')


def makeDistanceMatrix(n):
    logging.info('start')
    d = np.arange(n, dtype=int)
    dmat = np.repeat(d.reshape(1,n), n, axis=0) - d.reshape(n,1)
    dmat[dmat<0] = np.abs(dmat[dmat<0])
    logging.info('done')
    return dmat


def calcDistanceDecay(matrix, dmat):
    logging.info('start')
    n = matrix.shape[0]
    k = np.tri(n, dtype=bool)
    d = np.arange(n, dtype=int)
    x0 = dmat[~k]
    y0 = matrix[~k]
    dd = np.bincount(x0, weights=y0) / np.arange(n, 0, -1, dtype=int)
    logging.info('done')
    return dd


def fitSpline(dd, res, force_monotonic=False, knots=[5e3, 1e4 ,2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6]):
    knts = np.array(knots)
    knts = knts[knts >= res] / res
    y0 = dd
    x0 = np.arange(len(dd))
    y = y0[2:]
    x = x0[2:]
    k = x <= max(knots) / res
    spl = LSQUnivariateSpline(x[k], np.log10(y[k]+1e-6), t=knts, k=3, ext=0)
    z = 10**(spl(x0) - 1e-6)
    z[1:10] = dd[1:10]
    if dd[0] > dd[1]:
        z[0] = dd[0]
    else:
        z[0] = dd[1]
    if force_monotonic == 'non-increasing':
        zmin = z.min()
        kmin = np.where(z == zmin)[0][0]
        z[kmin:] = zmin
    elif force_monotonic == 'decreasing':
        zmin = z.min()
        kmin = np.where(z == zmin)[0][0]
        print(kmin)
        if kmin < 1:
            raise ValueError('min contact frequency found at min distance')
        elif kmin == z.size - 1:
            pass
        else:
            slope = (np.log10(z[kmin]) - np.log10(z[kmin-2])) / (x0[kmin] - x0[kmin-2])
            z[kmin+1:] = 10**(np.log10(z[kmin]) + slope * np.arange(len(dd) - kmin - 1))
    return z


def main(args):
    logging.info(args)
    res = int(args['r'])
    if args['F']:
        force_monotonic = 'decreasing'
    elif args['f']:
        force_monotonic = 'non-increasing'
    else:
        force_monotonic = False
    cmapFn = args['cmap']
    ddbgFn = args['ddbg']

    matrix = scipy.sparse.load_npz(cmapFn).toarray()
    matrix[np.isnan(matrix)] = 0
    n = matrix.shape[0]
    dmat = makeDistanceMatrix(n)
    dd = calcDistanceDecay(matrix, dmat)

    np.savetxt(ddbgFn, fitSpline(dd, res, force_monotonic=force_monotonic))
    logging.info('Done')
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
