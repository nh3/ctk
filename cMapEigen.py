#!/usr/bin/env python
'''
Usage: cMapEigen.py [options] <input> <outprfx>

Options:
    --corr      calculate based on correlation
    --plot      make plot
    --sm <int>  smooth window size [default: 51]
    <input>     input sparse matrix
    <outprfx>   output prefix, append "." when appropriate [default: ""]
    --debug     set logging level to DEBUG
    --prof      print performance profiling information
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import deque
from bisect import insort, bisect_left
from itertools import islice


def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result

def clean_mat(mat):
    amat = mat.toarray()
    s = np.nansum(amat, axis=0)
    ms = np.nanmean(s)
    ss = np.nanstd(s)
    d = np.nanstd(amat, axis=0)
    md = np.nanmean(d)
    sd = np.nanstd(d)
    k0 = np.where(np.logical_and(np.logical_and(s>ms-3*ss, s<ms+3*ss), np.logical_and(d>md-3*sd, d<md+3*md)))[0]
    mat = mat[k0][:,k0]
    return mat,k0

def plot_eigen(mat, V, filename, sm_width=51):
    m,n = V.shape
    hr = np.repeat((n,1),(1,n))
    fig = plt.figure(figsize=(4*n,4*np.sum(hr)))
    gs = gridspec.GridSpec(n+1,1, height_ratios=hr)
    ax = plt.subplot(gs[0])
    ax.imshow(mat, vmin=np.percentile(mat,2.5), vmax=np.percentile(mat,97.5), aspect='auto')
    for i in xrange(n):
        ax = plt.subplot(gs[i+1])
        y = V[:,i]
        y_rm = running_median_insort(y, sm_width)
        y_res = y - y_rm
        #ymin = np.percentile(y, 0.25)
        #ymax = np.percentile(y, 99.75)
        #ax.plot(y_res)
        ax.bar(range(m), y_res)
        #ax.set_ylim([ymin,ymax])
        plt.margins(x=0,y=0)
    plt.tight_layout()
    plt.savefig(filename)


def main(args):
    logging.info(args)
    smooth_width = int(args['sm'])
    if args['outprfx'] != '' and not args['outprfx'].endswith('.'):
        args['outprfx'] += '.'
    mat = scipy.sparse.load_npz(args['input']).astype(float)
    mat.data[np.isnan(mat.data)] = 0.0
    N = mat.shape[0]
    # set diagnal to 0
    kd = np.arange(N)
    mat[kd,kd] = 0.0
    # remove abnormal rows/columns
    cmat,k0 = clean_mat(mat)
    logging.info('dimension after cleaning: {}'.format(cmat.shape))
    # calculate eigen vectors
    if args['corr']:
        cmat = np.corrcoef(cmat.toarray())
        n = cmat.shape[0]
        cmat[np.arange(n),np.arange(n)] = 0.0
        v,V = scipy.linalg.eigh(cmat, eigvals=(n-5,n-1))
    else:
        v,V = scipy.sparse.linalg.eigsh(cmat, k=5)
    k = np.argsort(-abs(v))
    v = v[k]
    V0 = np.zeros((N,5))*np.nan
    V0[k0,] = V
    V0 = V0[:,k]
    V0[np.isnan(V0)] = np.tile(np.median(V, axis=0), N-len(k0))
    # save and plot
    np.savetxt(args['outprfx']+'eigenValue.txt', v, fmt='%.4e', delimiter='\t')
    np.savetxt(args['outprfx']+'eigenVector.txt', V0, fmt='%.4e', delimiter='\t')
    if args['plot']:
        if not args['corr']:
            mat = mat.toarray()
        plot_eigen(mat, V0, args['outprfx']+'eigenVector.pdf', sm_width=smooth_width)


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
