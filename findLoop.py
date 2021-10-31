#!/usr/bin/env python
'''
Call significant contact loops

Usage: findLoop.py -g <chrLen> [options] <bed> <cisBam> <transBam>

Options:
    -g <chrLen>     chromosome lengths
    -r <reg>        genomic region in "chrom[:start-end]" format
    -d <minD>       minimum distance [default: 1000]
    -D <maxD>       maximum distance [default: 1000000]
    -o <output>     output contact table, print to stdout if unspecified
    -O <outbin>     output bin table
    -p <float>      adjusted p value threshold [default: 0.05]
    -n <int>        minimum read pair count threhold [default: 5]
    -P <num>        cis coverage percentile above which bins are defined as peaks [default: 90]
    -C <num>        cis coverage percentile below which bins are discarded [default: 10]
    --ddprfx <str>  distance decay output prefix
    --bias <fn>     per bin bias as a one-column table matching <bed>, override --offpeak, --trans, --cpow
    --covexp <flt>  exponential of coverage to use as bin bias [default: 0.87]
    --offpeak       use offpeak coverage to calculate bias
    --trans         add trans coverage to calculate bias
    <bed>           sorted genomic intervals in bed format, minimum 4 columns
    <cisBam>        sorted bam of cis informative read pairs
    <transBam>      sorted bam of trans informative read pairs
    --debug         print debug info
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import sys
import os
import subprocess as sbp
import tempfile
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from scipy.interpolate import UnivariateSpline,LSQUnivariateSpline
from scipy.stats import binom,nbinom
from scipy.special import psi,gammaln
from scipy.optimize import fsolve
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.discrete.discrete_model import NegativeBinomial
pd.options.mode.chained_assignment = None


def read_chrom_length(fn):
    chrom_len = dict()
    with open(fn) as fh:
        for line in fh:
            fields = line.rstrip().split()
            chrom_len[fields[0]] = int(fields[1])
    return chrom_len

def parse_region(region, chrLen):
    chrom,sep,coords = region.partition(':')
    if sep == '':
        start = 0
        end = chrLen[chrom]
    else:
        start,sep,end = coords.partition('-')
        assert sep == '-', '{}: format error, expect "chrom:start-end"'.format(region)
        start = min(int(start), chrLen[chrom])
        end = min(int(end), chrLen[chrom])
    return chrom,start,end


def prepare_bed(inBed, parsed_region=None):
    tmp_bed,tmpBed = tempfile.mkstemp(suffix='.bed')
    if parsed_region is not None:
        reg_c,reg_s,reg_e = parsed_region
        cmd = '''awk '$1=="{}" && $2>={} && $3<={}' {} | cut -f1-4 > {}'''.format(reg_c, reg_s, reg_e, inBed, tmpBed)
    else:
        cmd = 'cut -f1-4 {} > {}'.format(inBed, tmpBed)
    logging.debug(cmd)
    sbp.call(cmd, shell=True, executable='/bin/bash')
    return tmpBed


def cis_to_contact(inBed, inBam, region=None):
    logging.debug('start')
    if region is not None:
        in_cmd = 'samtools view -u {} {} | bedtools intersect -a {} -b stdin -sorted -wo'.format(inBam, region, inBed)
    else:
        in_cmd = 'bedtools intersect -a {} -b {} -sorted -wo'.format(inBed, inBam)
    in_pipe = sbp.Popen(in_cmd, shell=True, executable='/bin/bash', stdout=sbp.PIPE)
    out_cmd = 'sort -k1,1V -k2,2V | bedtools groupby -g 1,2 -c 3 -o count'

    RPs = dict()
    for line in in_pipe.stdout:
        fields = line.rstrip().split()
        rname = fields[7]
        ovlp = int(fields[10])
        rpname,sep,i = rname.partition('/') # i indicates read1 or read2
        if sep is '':
            continue
        i = int(i)-1 # i = 0 -> read1; i = 1 -> read2
        if rpname not in RPs:
            RPs[rpname] = [None,None,0,0]
        elif ovlp < RPs[rpname][i+2]:
            continue
        RPs[rpname][i] = fields[3]
        RPs[rpname][i+2] = ovlp

    contacts = dict()
    for rpname in RPs:
        rp = RPs[rpname]
        if rp[0] is not None and rp[1] is not None:
            if rp[0] not in contacts:
                contacts[rp[0]] = dict()
            if rp[1] not in contacts[rp[0]]:
                contacts[rp[0]][rp[1]] = 0
            contacts[rp[0]][rp[1]] += 1
            if rp[1] not in contacts:
                contacts[rp[1]] = dict()
            if rp[0] not in contacts[rp[1]]:
                contacts[rp[1]][rp[0]] = 0
            contacts[rp[1]][rp[0]] += 1

    return contacts


def bed_coverage(inBed, inBam):
    logging.debug('start')
    cmd = 'sambamba depth region -t 8 -L {} {}'.format(inBed, inBam)
    pipe = sbp.Popen(cmd, executable='/bin/bash', shell=True, stdout=sbp.PIPE, stderr=open(os.devnull,'w'))

    coverage = dict()
    for line in pipe.stdout:
        if line.startswith('#'):
            continue
        fields = line.rstrip().split()
        name = fields[3]
        depth = int(fields[4])
        coverage[name] = depth

    return coverage


def make_bin_table(inBed, cisBam, transBam=None):
    logging.debug('start')
    bin_df = pd.read_table(inBed, sep='\t', header=None, names=['chrom','start','end','name'],
            dtype={'chrom':str,'start':int,'end':int,'name':str})
    bin_df['idx'] = range(len(bin_df))
    bin_df['length'] = (bin_df.end-bin_df.start).astype(int)
    bin_df['mid'] = ((bin_df.start+bin_df.end)/2).astype(int)
    cis_coverage = bed_coverage(inBed, cisBam)
    bin_df['cis'] = [cis_coverage[name] for name in bin_df['name']]
    if transBam is not None:
        trans_coverage = bed_coverage(inBed, transBam)
        bin_df['trans'] = [trans_coverage[name] for name in bin_df['name']]
    else:
        bin_df['trans'] = np.nan
    bin_df.set_index('name', drop=False, inplace=True)
    return bin_df


def mad(x):
    x0 = np.nanmedian(x)
    return np.nanmedian(np.abs(x - x0))


def define_peaks(bin_df, percent=90, nmad=None):
    if nmad is not None:
        x = np.nanmedian(bin_df.cis) + nmad * mad(bin_df.cis)
    else:
        x = np.percentile(bin_df.cis, percent)
    bin_df['peak'] = bin_df.cis > x
    return bin_df


def filter_bins(bin_df, percent=10):
    bin_df = bin_df[bin_df.cis > np.percentile(bin_df.cis, percent)]
    return bin_df


def make_contact_table(contacts, bin_df):
    logging.debug('start')
    ct_df = pd.DataFrame([[r1,r2,contacts[r1][r2]] for r1 in contacts for r2 in contacts[r1]],columns=['end1','end2','n'])
    ct_df['n1'] = np.where(np.less_equal(ct_df.end1, ct_df.end2), ct_df.end1, ct_df.end2)
    ct_df['n2'] = np.where(np.less_equal(ct_df.end1, ct_df.end2), ct_df.end2, ct_df.end1)
    ct_df.end1 = ct_df.n1
    ct_df.end2 = ct_df.n2
    ct_df.drop(['n1','n2'], axis=1, inplace=True)
    ct_df.drop_duplicates(inplace=True)
    ct_df = pd.merge(
            bin_df[['name','idx','length','mid','trans','cis','bias']],
            pd.merge(bin_df[['name','idx','length','mid','trans','cis','bias']], ct_df, how='inner', left_on='name', right_on='end2'),
            how='inner', left_on='name', right_on='end1', suffixes=('1','2'))
    ct_df.drop(['end1','end2'], axis=1, inplace=True)
    ct_df['d'] = abs(ct_df.idx1 - ct_df.idx2)
    ct_df['D'] = abs(ct_df.mid1 - ct_df.mid2)
    return ct_df


def fit_distance_function(contact_df, bin_df, outprfx=None):
    logging.debug('start')
    dd = contact_df[['d','D','n']]
    dd = dd.loc[dd.d>=2]
    dd['x'] = dd.D
    N = len(bin_df)
    n = dd.groupby(by='d')['x'].count()
    x = np.log10(dd.groupby(by='d')['x'].mean())
    y = dd.groupby(by='d')['n'].sum() / (N - np.unique(np.sort(dd.d)))
    k = np.logical_and(y > 0, x < np.percentile(x, 80))
    knts = np.log10(np.array([3e3,4e3,5e3,6e3,8e3,1e4,2e4,5e4,1e5,2e5,5e5,1e6]))
    spl_fail = True
    while spl_fail:
        try:
            spl = LSQUnivariateSpline(x[k].values, np.log10(y[k].values), t=knts, k=3, ext=3)
            spl_fail = False
        except ValueError:
            logging.warn('LSQUnivariateSpline() failed, shrink knt')
            knts = knts[1:]
    if outprfx is not None:
        pd.DataFrame({'x':x, 'y':spl(x)}).to_csv(outprfx+'.txt', sep='\t', float_format='%.2e', index=False, header=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.loglog(10**x[k],y[k],'o', 10**x,10**spl(x),'-')
        plt.grid(True)
        plt.savefig(outprfx+'.png', format='png', bbox_inches='tight')
    return spl


def calc_bias(bin_df, contacts, offpeak=False, include_trans=False, covexp=1):
    logging.debug('start')
    if offpeak:
        cov_bias = np.zeros(len(bin_df))
        peak_names = bin_df['name'][bin_df.peak]
        for i,r1 in enumerate(bin_df.name):
            x = 1
            if r1 in contacts:
                for r2 in contacts[r1]:
                    if r2 in peak_names:
                        continue
                    x += contacts[r1][r2]
            cov_bias[i] = x
            if include_trans:
                cov_bias[i] += bin_df.trans[i]
    else:
        cov_bias = bin_df.cis + 1
        if include_trans:
            cov_bias += bin_df.trans
    cov_bias = cov_bias**covexp
    cov_bias = cov_bias / np.nanmedian(cov_bias)
    bin_df['bias'] = cov_bias
    return bin_df


def calc_baseline(bin_df, chrom_len, thisChrom):
    L = sum([chrom_len[c] for c in chrom_len if c != thisChrom])
    m = len(bin_df) * float(L) / chrom_len[thisChrom]
    bin_df['base'] = bin_df.trans / m
    return bin_df


def call_loop(contact_df, nTransBin, distFunc, probFunc, minD=1000, maxD=1000000):
    logging.debug('start')
    contact_df['bias'] = contact_df.bias1 * contact_df.bias2
    contact_df['df'] = np.zeros(len(contact_df))
    contact_df['expected'] = np.zeros(len(contact_df))
    contact_df['p'] = np.ones(len(contact_df))
    contact_df['padj'] = np.ones(len(contact_df))
    k_d = np.logical_and(contact_df.D>=minD, contact_df.D<=maxD)
    N = np.sum(contact_df.n[k_d])
    logging.debug(N)
    n = contact_df.n[k_d].values.astype(int)
    D = contact_df.D[k_d].values
    b1 = contact_df.bias1[k_d].values
    b2 = contact_df.bias2[k_d].values
    df = 10**distFunc(D)
    baseline = np.minimum(contact_df.trans1[k_d].values, contact_df.trans2[k_d].values)/float(nTransBin)
    u = (b1*b2*df+baseline).astype(float)/N
    p = np.ones(len(n))
    k_u = u > 0
    p[k_u] = 1.0-probFunc(n[k_u]-1, N, u[k_u])
    padj = multipletests(p, method='holm')[1]
    contact_df.at[k_d, 'df'] = df
    contact_df.at[k_d, 'expected'] = u*N
    contact_df.at[k_d, 'p'] = p
    contact_df.at[k_d, 'padj'] = padj
    return contact_df


def main(args):
    logging.debug(args)
    inBed = args['bed']
    cisBam = args['cisBam']
    transBam = args['transBam']
    chrLenFn = args['g']
    region = args['r']
    minD = int(args['d'])
    maxD = int(args['D'])
    peakPct = float(args['P'])
    covPct = float(args['C'])
    maxP = float(args['p'])
    minRP = int(args['n'])
    outctct = args['o']
    outbin = args['O']
    ddprfx = args['ddprfx']
    biasFn = args['bias']
    covexp = float(args['covexp'])
    offpeak = args['offpeak']
    add_trans=args['trans']

    chrLen = read_chrom_length(chrLenFn)
    reg_c,reg_s,reg_e = parse_region(region, chrLen)
    transL = sum([chrLen[c] for c in chrLen if c != reg_c])

    tmpBed = prepare_bed(inBed, [reg_c,reg_s,reg_e])
    try:
        bin_df = make_bin_table(tmpBed, cisBam, transBam)
        bin_df = define_peaks(bin_df, percent=peakPct)
        contacts = cis_to_contact(tmpBed, cisBam, region)
        if biasFn is not None:
            bias = np.loadtxt(biasFn)
            bin_df['bias'] = 1/(bias/np.nanmedian(bias))
        else:
            bin_df = filter_bins(bin_df, covPct)
            bin_df = calc_bias(bin_df, contacts, offpeak=offpeak, include_trans=add_trans, covexp=covexp)
        contact_df = make_contact_table(contacts, bin_df)
        d_f = fit_distance_function(contact_df, bin_df, outprfx=ddprfx+'.pass1')
        dF = np.frompyfunc(lambda x: d_f(np.log10(x)), 1, 1)
        n_trans_bin = len(bin_df) * float(transL) / (reg_e - reg_s)
        contact_df = call_loop(contact_df, n_trans_bin, distFunc=dF, probFunc=binom.cdf, minD=minD, maxD=maxD)
        k = np.logical_and(contact_df.padj < maxP, contact_df.n >= minRP)
        d_f2 = fit_distance_function(contact_df[~k], bin_df, outprfx=ddprfx+'.pass2')
        dF2 = np.frompyfunc(lambda x: d_f2(np.log10(x)), 1, 1)
        contact_df = call_loop(contact_df, n_trans_bin, distFunc=dF2, probFunc=binom.cdf, minD=minD, maxD=maxD)
        if outctct is None:
            outctct = sys.stdout
        contact_df.to_csv(outctct, sep='\t', header=True, index=False)
        if outbin is not None:
            bin_df.to_csv(outbin, sep='\t', header=True, index=False)
    finally:
        os.remove(tmpBed)


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
                format='%(asctime)s; %(levelname)s; %(funcName)s(); %(message)s',
                datefmt='%y-%m-%d %H:%M:%S')
        if args.get('prof'):
            import cProfile
            cProfile.run('main(args)')
        else:
            main(args)
    except KeyboardInterrupt:
        logging.warning('Interrupted')
        sys.exit(1)
