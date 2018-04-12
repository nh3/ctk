#!/usr/bin/env python
'''
Usage: findLoop.py [options] <bed> <cisBam> <transBam> <validBam>

Options:
    -r <reg>    genomic region in "chrom:start-end" format
    -p <num>    valid coverage percentile above which bins are defined as peaks [default: 90]
    -c <num>    cis coverage percentile below which bins are discarded [default: 10]
    <bed>       sorted genomic intervals in bed format, minimum 4 columns
    <cisBam>    sorted bam of cis informative read pairs
    <transBam>  sorted bam of trans informative read pairs
    <validBam>  sorted bam of valid read pairs
    --debug     print debug info
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import os
import os.path
import subprocess as sbp
import tempfile
from scipy.sparse import dok_matrix
from scipy.interpolate import UnivariateSpline
from scipy.stats import nbinom
from scipy.special import psi,gammaln
from scipy.optimize import fsolve
import numpy as np
import pandas as pd

def parse_region(region):
    chrom,sep,coords = region.partition(':')
    assert sep is not None, '{}: format error, expect "chrom:start-end"'.format(region)
    start,sep,end = coords.partition('-')
    assert sep is not None, '{}: format error, expect "chrom:start-end"'.format(region)
    return chrom,int(start),int(end)


def prepare_bed(inBed, region=None):
    tmp_bed,tmpBed = tempfile.mkstemp(suffix='.bed')
    if region is not None:
        reg_c,reg_s,reg_e = parse_region(region)
        cmd = '''awk '$1=="{}" && $2>={} && $3<={}' {} | cut -f1-4 > {}'''.format(reg_c, reg_s, reg_e, inBed, tmpBed)
    else:
        cmd = 'cut -f1-4 {} > {}'.format(inBed, tmpBed)
    logging.debug(cmd)
    sbp.call(cmd, shell=True, executable='/bin/bash')
    return tmpBed


def cis_to_contact(inBed, inBam, region=None):
    if region is not None:
        in_cmd = 'samtools view -u {} {} | bedtools intersect -a {} -b stdin -sorted -wo'.format(inBam, region, inBed)
    else:
        in_cmd = 'bedtools intersect -a {} -b {} -sorted -wo'.format(inBed, inBam)
    logging.debug(in_cmd)
    in_pipe = sbp.Popen(in_cmd, shell=True, executable='/bin/bash', stdout=sbp.PIPE)
    out_cmd = 'sort-alt -k1,1N -k2,2N | bedtools groupby -g 1,2 -c 3 -o count'

    RPs = dict()
    for line in in_pipe.stdout:
        fields = line.rstrip().split()
        rname = fields[7]
        ovlp = int(fields[10])
        rpname,sep,i = rname.partition('/')
        if sep is '':
            continue
        i = int(i)-1
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
    cmd = 'sambamba depth region -t 8 -L {} {}'.format(inBed, inBam)
    logging.debug(cmd)
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


def make_bin_table(inBed, valid_coverage, trans_coverage, cis_coverage):
    bin_df = pd.read_table(inBed, sep='\t', header=None, names=['chrom','start','end','name'],
            dtype={'chrom':str,'start':int,'end':int,'name':str})
    bin_df['idx'] = range(len(bin_df))
    bin_df['length'] = (bin_df.end-bin_df.start).astype(int)
    bin_df['mid'] = ((bin_df.start+bin_df.end)/2).astype(int)
    bin_df['valid'] = [valid_coverage[name] for name in bin_df['name']]
    bin_df['trans'] = [trans_coverage[name] for name in bin_df['name']]
    bin_df['cis'] = [cis_coverage[name] for name in bin_df['name']]
    bin_df.set_index('name', drop=False, inplace=True)
    return bin_df


def define_peaks(bin_df, percent=90):
    bin_df['peak'] = bin_df.valid > np.percentile(bin_df.valid, percent)
    return bin_df


def filter_bins(bin_df, percent=10):
    bin_df = bin_df[bin_df.cis > np.percentile(bin_df.cis, percent)]
    return bin_df


def make_contact_table(contacts, bin_df):
    ct_df = pd.DataFrame([[r1,r2,contacts[r1][r2]] for r1 in contacts for r2 in contacts[r1]],columns=['end1','end2','n'])
    ct_df['n1'] = np.where(np.less_equal(ct_df.end1, ct_df.end2), ct_df.end1, ct_df.end2)
    ct_df['n2'] = np.where(np.less_equal(ct_df.end1, ct_df.end2), ct_df.end2, ct_df.end1)
    ct_df.end1 = ct_df.n1
    ct_df.end2 = ct_df.n2
    ct_df.drop(['n1','n2'], axis=1, inplace=True)
    ct_df.drop_duplicates(inplace=True)
    ct_df = pd.merge(
            bin_df[['name','idx','length','mid','valid','trans','cis','offpeak']],
            pd.merge(bin_df[['name','idx','length','mid','valid','trans','cis','offpeak']], ct_df, how='inner', left_on='name', right_on='end2'),
            how='inner', left_on='name', right_on='end1', suffixes=('1','2'))
    ct_df.drop(['end1','end2'], axis=1, inplace=True)
    ct_df['d'] = abs(ct_df.idx1 - ct_df.idx2)
    ct_df['D'] = abs(ct_df.mid1 - ct_df.mid2)
    return ct_df


def fit_distance_function(contact_df, bin_df):
    dd = contact_df[['d','D','n']]
    dd = dd.loc[dd.D>1e3]
    dd['x'] = np.log10(dd.D)
    dd['y'] = np.log10(dd.n)
    N = len(bin_df)
    n = dd.groupby(by='d')['x'].count()
    x = dd.groupby(by='d')['x'].mean()
    y = dd.groupby(by='d')['y'].sum() / (N - np.unique(np.sort(dd.d)))
    spl = UnivariateSpline(x, y)
    return spl


def make_contact_matrix(contact_df, bin_df):
    n = len(bin_df)
    mat = dok_matrix((n,n), dtype=float)
    for r1 in contacts:
        i1 = bin_df.loc[r1, 'idx']
        for r2 in contacts[r1]:
            i2 = bin_df.loc[r2, 'idx']
            mat[i1,i2] = contacts[r1][r2]
            mat[i1,i2] = contacts[r1][r2]
    return mat.to_csr()


def calc_offpeak_coverage(bin_df, contacts):
    bin_df['offpeak'] = np.zeros(len(bin_df))
    peak_names = bin_df['name'][bin_df.peak]
    for r1 in bin_df.name:
        if r1 not in contacts:
            continue
        x = 1
        for r2 in contacts[r1]:
            if r2 in peak_names:
                continue
            x += contacts[r1][r2]
        bin_df.at[r1, 'offpeak'] = x
    return bin_df


import mpmath

class negBin(object):
    def __init__(self, p=0.1, r=10):
        nbin_mpmath = lambda k, p, r: mpmath.gamma(k + r)/(mpmath.gamma(k+1)*mpmath.gamma(r))*np.power(1-p, r)*np.power(p, k)
        self.nbin = np.frompyfunc(nbin_mpmath, 3, 1)
        self.p = p
        self.r = r

    def mleFun(self, par, data):
        '''
        Objective function for MLE estimate according to
        https://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation

        Keywords:
        data -- the points to be fit
        '''
        p = par[0]
        r = par[1]
        n = len(data)
        m = np.mean(data)
        f0 = m/(r+m)-p
        f1 = np.sum(psi(data+r)) - n*psi(r) + n*np.log(r/(r+m))
        return np.array([f0, f1])

    def fit(self, data, p=None, r=None):
        if p is None or r is None:
            mu = np.mean(data)
            sigma = np.std(data)
            r = (mu*mu)/(sigma*sigma-mu)
            p = (sigma*sigma-mu)/(sigma*sigma)
        x = fsolve(self.mleFun, np.array([p, r]), args=(data,))
        self.p = x[0]
        self.r = x[1]

    def pdf(self, k):
        return self.nbin(k, self.p, self.r).astype('float64')
    

def main(args):
    logging.debug(args)
    inBed = args['bed']
    cisBam = args['cisBam']
    transBam = args['transBam']
    validBam = args['validBam']
    region = args['r']
    peakPct = float(args['p'])
    covPct = float(args['c'])

    tmpBed = prepare_bed(inBed, region)
    try:
        cis_coverage = bed_coverage(tmpBed, cisBam)
        trans_coverage = bed_coverage(tmpBed, transBam)
        valid_coverage = bed_coverage(tmpBed, validBam)
        contacts = cis_to_contact(tmpBed, cisBam, region)
        bin_df = make_bin_table(tmpBed, valid_coverage, trans_coverage, cis_coverage)
        bin_df = filter_bins(define_peaks(bin_df, peakPct), covPct)
        bin_df = calc_offpeak_coverage(bin_df, contacts)
        contact_df = make_contact_table(contacts, bin_df)
        fD = fit_distance_function(contact_df, bin_df)
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
