#!/usr/bin/env python
'''
Combine chromosomal APA into genomewide APA

Usage: combineRmbgAPA2D.py [options] <path> <outprfx>

Options:
    --maxF <maxF>   max fold change represented by color range [default: 1.5]
    --cRange <v>    max color range in percentile [default: 99]
    --debug         run in debug mode
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
from scipy.stats import ttest_ind
from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap


def plotHeatmap(matMean, outpdf, minPct=2, maxPct=98, maxFC=1.5):
    RdWh = LinearSegmentedColormap.from_list('RdWh', [(0,'white'),(1,'red')])
    xmin = np.nanpercentile(matMean, minPct)
    pp = PdfPages(outpdf)
    plt.imshow(np.log2(matMean), cmap=RdWh, vmin=np.log2(xmin), vmax=np.log2(xmin*maxFC))
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    plt.close()
    plt.imshow(matMean, cmap=RdWh, vmin=xmin, vmax=np.percentile(matMean, maxPct))
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    plt.close()
    sm = gaussian_filter(matMean, 1, mode='nearest')
    plt.imshow(sm, cmap=RdWh, vmin=xmin, vmax=xmin*maxFC)
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    plt.close()
    pp.close()


def main(args):
    logging.info(args)
    path = args['path']
    outprfx = args['outprfx']
    maxF = float(args['maxF'])
    cRange = float(args['cRange'])
    outFn = outprfx + '.rmbgAPA.stat'
    outPdf = outprfx + '.rmbgAPA.pdf'

    byChrom = []
    for item in os.listdir(path):
        fpath = os.path.join(path, item)
        if os.path.isdir(fpath) and item.startswith('chr'):
            byChrom.append(fpath)

    matFns = [os.path.join(p, 'rmbgAPA.txt') for p in byChrom]
    statFns = [os.path.join(p, 'rmbgAPA.stat') for p in byChrom]
    mats = [np.loadtxt(f) for f in matFns]
    stats = [pd.read_table(f,header=None)[[3,4,5,6,7,8,9,10,11,12]].as_matrix() for f in statFns]
    Ns = [x.shape[0] for x in stats]
    matSum = np.nansum(np.array([n*mat for n,mat in zip(Ns, mats)]), axis=0)
    matMean = matSum / sum(Ns)
    plotHeatmap(matMean, outPdf, maxPct=cRange, maxFC=maxF)
    stat = np.vstack(stats)
    stat = stat[~np.logical_or(np.isinf(stat).any(axis=1), np.isnan(stat).any(axis=1))]
    k = np.sum(stat>0, axis=1) >= 5
    stat = stat[k]

    center = stat[:,0]
    bg = stat[:,1]
    TLBR = stat[:,2:6]
    diag_corners = stat[:,(6,8)]
    offdiag_corners = stat[:,(7,9)]
    fc_bg = np.nanmean(center)/np.nanmean(bg)
    t_bg,p_bg = ttest_ind(center, bg, equal_var=False, nan_policy='omit')
    fc_tlbr = np.nanmean(center)/np.nanmean(TLBR)
    t_tlbr,p_tlbr = ttest_ind(center, np.nanmean(TLBR,axis=1), equal_var=False, nan_policy='omit')
    fc_onoff = np.nanmean(diag_corners)/np.nanmean(offdiag_corners)
    t_onoff,p_onoff = ttest_ind(np.nanmean(diag_corners,axis=1), np.nanmean(offdiag_corners,axis=1), equal_var=False, nan_policy='omit')
    with open(outFn, 'w') as fh:
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(path, sum(Ns), fc_bg, p_bg, fc_tlbr, p_tlbr, fc_onoff, p_onoff), file=fh)


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
