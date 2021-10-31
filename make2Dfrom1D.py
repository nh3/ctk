#!/usr/bin/env python
'''
Make 2D contacts from 1D features

Usage: make2Dfrom1D.py domain [options] -g <chrL> <input1>
       make2Dfrom1D.py (loop|compartment) [options] -g <chrL> <input1> [<input2>]

Options:
    <input1>    input 1D features in BED format
    <input2>    input 1D features in BED format, if given, contacts are made between <input1> and <input2>
    -g <chrL>   chromomsome lengths
    -r <res>    resolution in bp [default: 1000]
    -f <flkL>   flanking size in bp [default: 0]
    -m <mode>   method to extract flanking regions: fixed, scaled, or truncated [default: truncated]
    -d <minD>   minimum distance [default: 10000]
    -D <maxD>   maximum distance [default: 1000000]
    --debug     print debug info
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import numpy as np
import pandas as pd

def read_features(filename):
    features = pd.read_csv(filename, sep='\t', header=None)
    features.rename(columns={0:'chrom',1:'start',2:'end'}, inplace=True)
    essential = features.loc[:,'chrom':'end']
    extra = features.iloc[:,3:]
    essential['mid'] = np.round((essential.start+essential.end)/2).astype(int)
    return essential,extra

def read_chrom_length(filename):
    chrLen = pd.read_csv(filename, sep='\t', names=['length'], index_col=0).to_dict()['length']
    return chrLen

def get_neighbour_boundary(features, chrLen):
    features['prev_e'] = 0
    features['next_s'] = 0
    for chrom in np.unique(features.chrom.values):
        L = chrLen[chrom]
        I = features.index[features.chrom == chrom]
        i = min(I)
        j = max(I)
        features.loc[(i+1):j, 'prev_e'] = features.loc[i:(j-1), 'end'].values
        features.loc[i:(j-1), 'next_s'] = features.loc[(i+1):j, 'start'].values
        features.loc[j, 'next_s'] = L
    logging.debug('done')
    return features

def get_center_boundary(features, chrLen, res, focal=True):
    if focal:
        features['cent_s'] = np.round(features.mid-res/2).astype(int)
        features['cent_e'] = np.round(features.mid+res/2).astype(int)
    else:
        features['cent_s'] = features.start
        features['cent_e'] = features.end
    features.loc[features.cent_s < 0, 'cent_s'] = 0
    for chrom in np.unique(features.chrom.values):
        L = chrLen[chrom]
        features.loc[(features.chrom == chrom) & (features.cent_e > L), 'cent_e'] = L
    logging.debug('done')
    return features

def get_flanking(features, chrLen, w=None, mode='truncated'):
    assert w is not None or mode == 'scaled', '"-f" required when "-m" is not "scaled"'
    if mode == 'truncated':
        features['flnk_s'] = np.maximum(features.cent_s-w, features.prev_e)
        features['flnk_e'] = np.minimum(features.cent_e+w, features.next_s)
    elif mode == 'scaled':
        features['flnk_s'] = features.prev_e
        features['flnk_e'] = features.next_s
    elif mode == 'fixed':
        features['flnk_s'] = features.cent_s-w
        features['flnk_e'] = features.cent_e+w
    else:
        raise ValueError('{}: unsupported mode'.format(mode))
    features.loc[features.flnk_s < 0, 'flnk_s'] = 0
    for chrom in np.unique(features.chrom.values):
        L = chrLen[chrom]
        features.loc[(features.chrom == chrom) & (features.flnk_e > L), 'flnk_e'] = L
    logging.debug('done')
    return features

def print_contact(features1, extra1, features2=None, extra2=None, minD=0, maxD=1e8):
    logging.debug('start')
    chroms = np.unique(features1.chrom.values)
    if features2 is None:
        for chrom in chroms:
            k = np.where(features1.chrom == chrom)[0]
            F1 = features1.iloc[k].values
            E1 = extra1[k]
            n = len(k)
            for i in range(n):
                f1 = F1[i]
                e1 = E1[i]
                for j in range(i+1,n):
                    f2 = F1[j]
                    e2 = E1[j]
                    d = f2[3] - f1[3]
                    if d > minD and d < maxD:
                        dband = int((np.log10(d) - 3) * 2)
                        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                            chrom, f1[8], f1[9], chrom, f2[8], f2[9], 
                            chrom, f1[6], f1[7], chrom, f2[6], f2[7], dband, e1, e2))
    else:
        for chrom in chroms:
            k1 = np.where(features1.chrom == chrom)[0]
            F1 = features1.iloc[k1].values
            E1 = extra1[k1]
            k2 = np.where(features2.chrom == chrom)[0]
            F2 = features2.iloc[k2].values
            E2 = extra2[k2]
            n1 = len(F1)
            n2 = len(F2)
            for i in range(n1):
                f1 = F1[i]
                e1 = E1[i]
                for j in range(n2):
                    f2 = F2[j]
                    e2 = E2[j]
                    d = abs(f1[3] - f2[3])
                    if d > minD and d < maxD:
                        dband = int((np.log10(d) - 3) * 2)
                        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                            chrom, f1[8], f1[9], chrom, f2[8], f2[9], 
                            chrom, f1[6], f1[7], chrom, f2[6], f2[7], dband, e1, e2))
    logging.debug('done')

def print_self_contact(features):
    logging.debug('start')
    features['d'] = 0
    features[['chrom','flnk_s','flnk_e','chrom','flnk_s','flnk_e',
              'chrom','cent_s','cent_e','chrom','cent_s','cent_e','d']].to_csv(sys.stdout, sep='\t', header=False, index=False)
    logging.debug('done')


def main(args):
    logging.info(args)
    inputFn1 = args['input1']
    inputFn2 = args['input2']
    res = int(args['r'])
    flkLen = int(args['f'])
    flkMode = args['m']
    minD = int(args['d'])
    maxD = int(args['D'])

    chrLen = read_chrom_length(args['g'])

    features1,extra1 = read_features(inputFn1)
    if extra1.empty:
        extra1 = np.zeros(features1.shape[0], dtype=int) # set to zero
    else:
        extra1 = extra1.apply(lambda x: ('\t'.join(list(map(str, x)))), axis=1).values # convert to an array containing concatenated columns
    features1 = get_neighbour_boundary(features1, chrLen)
    features1 = get_center_boundary(features1, chrLen, res, focal=args['loop'])
    features1 = get_flanking(features1, chrLen, w=flkLen, mode=flkMode)

    features2,extra2 = None,None
    if inputFn2 is not None:
        features2,extra2 = read_features(inputFn2)
        if extra2.empty:
            extra2 = np.zeros(features2.shape[0], dtype=int) # set to zero
        else:
            extra2 = extra2.apply(lambda x: ('\t'.join(list(map(str, x)))), axis=1).values # convert to an array containing concatenated columns
        features2 = get_neighbour_boundary(features2, chrLen)
        features2 = get_center_boundary(features2, chrLen, res, focal=args['loop'])
        features2 = get_flanking(features2, chrLen, w=flkLen, mode=flkMode)
    else:
        features2 = None

    if args['domain']:
        print_self_contact(features1)
    else:
        print_contact(features1, extra1, features2=features2, extra2=extra2, minD=minD, maxD=maxD)


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
