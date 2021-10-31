#!/usr/bin/env python
'''
Calculate pearson and spearman correlation between two C-maps

Usage: cMapCorr.py [--data] <input1> <input2>

Options:
    --data      print data value instead of statistics 
    --debug     print debug info
    <input1>    input matrix 1
    <input2>    input matrix 2
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import sys
import numpy as np
import scipy.sparse
from scipy.stats import pearsonr,spearmanr

def get_flatten_idx(mat):
    n = mat.shape[0]
    m = len(mat.data)
    I = mat.indices + np.repeat(np.arange(n)*n, np.diff(mat.indptr))
    dict_I = {I[i]:i for i in xrange(m) if not np.isnan(mat.data[i])}
    return dict_I


def main(args):
    inFn1 = args['input1']
    inFn2 = args['input2']
    mat1 = scipy.sparse.load_npz(inFn1)
    mat2 = scipy.sparse.load_npz(inFn2)
    I1 = get_flatten_idx(mat1)
    I2 = get_flatten_idx(mat2)
    k = np.intersect1d(I1.keys(), I2.keys(), assume_unique=True)
    k1 = [I1.get(i, 0) for i in k]
    k2 = [I2.get(i, 0) for i in k]
    if args['data']:
        import pandas as pd
        odf = pd.DataFrame({'map1':mat1.data[k1],'map2':mat2.data[k2]})
        odf.to_csv(sys.stdout, sep='\t', columns=['map1','map2'], header=False, index=False)
    else:
        r,p = pearsonr(mat1.data[k1], mat2.data[k2])
        rho,p = spearmanr(mat1.data[k1], mat2.data[k2])
        print('{}\t{}\t{}\t{}\t{}'.format(inFn1, inFn2, len(k), rho, r))


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
