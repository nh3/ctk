#!/usr/bin/env python
'''
Usage: cMapReduceRes.py -f <fromRes> -t <toRes> <input> <output>

Options:
    -f <fromRes>    original (higer) resolution (e.g. 1000)
    -t <toRes>      target (lower) resolution, must be an integer multiply of "-f" (e.g. 5000)
    <input>         input higher resolution sparse matrix
    <output>        output lower resolution sparse matrix
    --debug         set logging level to DEBUG
    --prof          print performance profiling information
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def main(args):
    logging.info(args)
    fromRes = int(args['f'])
    toRes = int(args['t'])
    assert toRes % fromRes == 0, 'toRes is not an integer multiply of fromRes'
    z = toRes / fromRes
    if z == 1: # do nothing
        return 0
    mat = scipy.sparse.load_npz(args['input']).toarray()
    n = mat.shape[0]
    m = int(n/z)
    mm = n % z
    if mm == 0:
        new_mat = mat.reshape(m,z,m,z).sum(-1).sum(1)
    else:
        tmp_mat = np.zeros((z*(m+1),z*(m+1)), dtype=int)
        tmp_mat[0:n,0:n] = mat
        new_mat = tmp_mat.reshape(m+1,z,m+1,z).sum(-1).sum(1)
        del tmp_mat
    scipy.sparse.save_npz(args['output'], scipy.sparse.csr_matrix(new_mat))


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
