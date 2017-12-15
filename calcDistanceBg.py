#!/usr/bin/env python
'''
Calculate distance-decay profile for a chromosome or part of a chromosome

Usage: calcDistanceBg.py [options] <cmap>

Options:
    -o <output>     output
    -r <res>        resolution in number of bins [default: 1]
    -k <minK>       left boundary of the chosen region in number of bins [default: 0]
    -K <maxK>       right boundary of the chosen region in number of bins
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
import numpy as np
from scipy.sparse import load_npz,save_npz
import pandas as pd
from scipy.interpolate import PchipInterpolator


def getDistance(mat):
    n = mat.shape[0]
    d = np.concatenate(
            [mat.indices[mat.indptr[i]:mat.indptr[i+1]]-i for i in xrange(n)])
    return d


def makeDistanceMatrix(n, res):
    d = np.arange(n, dtype=float)/res
    dmat = np.abs(
            np.repeat(d.reshape(1,n), n, axis=0) - d.reshape(n,1)).astype(int)
    return dmat


def calcDistanceDecay(mat):
    n = mat.shape[0]
    d = getDistance(mat)
    k = d > 0
    x = d[k]
    y = mat.data[k]

    cutoffs = np.unique(np.percentile(x, np.arange(0,100,0.25)))
    cutoffs = np.append(cutoffs, n-1)
    xa = np.repeat(np.arange(n-1)+1, np.arange(n-1)[::-1]+1)
    xb = np.digitize(xa, cutoffs, right=True)
    xn = np.bincount(xb)
    xd = np.bincount(xb, weights=xa) / xn
    yd = np.bincount(np.digitize(x,cutoffs, right=True), weights=y) / xn
    return xd,yd,xn


def main(args):
    logging.info(args)
    res = int(args['r'])
    output = args['o']
    minK = int(args['k'])
    maxK = args['K']
    cmapFn = args['cmap']

    matrix = load_npz(cmapFn)
    matrix.data[np.isnan(matrix.data)] = 0
    n = matrix.shape[0]

    if maxK is None:
        maxK = n
    else:
        maxK = int(maxK)

    matrix = matrix[minK:maxK,minK:maxK]
    n = matrix.shape[0]

    xd,yd,xn = calcDistanceDecay(matrix)
    w = 1/np.sqrt(xn)
    w = w/np.max(w)
    f = PchipInterpolator(np.log1p(xd), np.log1p(yd))
    xf = np.arange(n-1)+1
    yf = np.expm1(f(np.log1p(xf)))

    if output is not None:
        try:
            if dirname(output):
                os.makedirs(dirname(output))
        except OSError as e:
            if e.errno == errno.EEXIST and isdir(dirname(output)):
                pass
            else:
                raise
    else:
        output = sys.stdout

    dd_tbl = pd.DataFrame({'d':xf, 'dd':yf})
    dd_tbl.to_csv(output, sep='\t', float_format='%.2e',
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
