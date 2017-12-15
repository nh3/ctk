#!/usr/bin/env python
'''
Combine chromosomal APA into genomewide APA

Usage: combineRmbgAPA.py [options] <path>

Options:
    --debug     run in debug mode
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import errno
import os.path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages                                                                                                                                           
from matplotlib.colors import LinearSegmentedColormap



def plotHeatmap(matMean, outpdf):
    n,m = matMean.shape
    x = n - int(n/2+1)
    y = m - int(m/2+1)
    RdWh = LinearSegmentedColormap.from_list('RdWh', [(0,'white'),(1,'red')])
    plt.imshow(matMean, cmap=RdWh, extent=[-x-0.5,n-x-0.5,-y-0.5,m-y-0.5])
    pp = PdfPages(outpdf)                                                                                                                                                                      
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()

def main(args):
    logging.info(args)
    path = args['path']
    outFn = os.path.join(args['path'], 'rmbgAPA.stat')
    outPdf = os.path.join(args['path'], 'rmbgAPA.pdf')

    byChrom = []
    for item in os.listdir(path):
        fpath = os.path.join(path, item)
        if os.path.isdir(fpath) and item.startswith('chr'):
            byChrom.append(fpath)

    matFns = [os.path.join(p, 'rmbgAPA.txt') for p in byChrom]
    statFns = [os.path.join(p, 'rmbgAPA.stat') for p in byChrom]
    mats = [np.loadtxt(f) for f in matFns]
    stats = [pd.read_table(f,header=None)[[3,4,5]].as_matrix() for f in statFns]
    Ns = [x.shape[0] for x in stats]
    matSum = np.sum(np.array([n*mat for n,mat in zip(Ns, mats)]), axis=0)
    matMean = matSum / sum(Ns)
    plotHeatmap(matMean, outPdf)
    stat = np.vstack(stats)
    stat = stat[~np.logical_or(np.isinf(stat).any(axis=1), np.isnan(stat).any(axis=1))]

    mids = stat[:,0]
    rest_means = stat[:,1]
    LL_means = stat[:,2]
    fc_rest = np.nanmean(mids)/np.nanmean(rest_means)
    t_rest,p_rest = ttest_ind(mids, rest_means, equal_var=False, nan_policy='omit')
    fc_LL = np.nanmean(mids)/np.nanmean(LL_means)
    t_LL,p_LL = ttest_ind(mids, LL_means, equal_var=False, nan_policy='omit')
    with open(outFn, 'w') as fh:
        print('{}\t{}\t{}\t{}\t{}\t{}'.format(path, sum(Ns), fc_rest, p_rest, fc_LL, p_LL),file=fh)


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.lstrip('-<').rstrip('>'):args[k] for k in args}
    try:
        if args['debug']:
            logLevel = logging.DEBUG
        else:
            logLevel = logging.WARN
        logging.basicConfig(
                level=logLevel,
                format='%(asctime)s; %(levelname)s; %(funcName)s; %(message)s',
                datefmt='%y-%m-%d %H:%M:%S')
        main(args)
    except KeyboardInterrupt:
        logging.warning('Interrupted')
        sys.exit(1)
