#!/usr/bin/env python
'''
Usage: make2Dfrom1D.py domain [options] -g <chrL> <input1>
       make2Dfrom1D.py (center|compartment) [options] -g <chrL> <input1> [<input2>]

Options:
    <input1>    input 1D features in BED format
    <input2>    input 1D features in BED format, if given, contacts are made between <input1> and <input2>
    -g <chrL>   chromomsome lengths
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

def read_features(filename):
    features = {}
    with open(filename) as fh:
        for line in fh:
            fields = line.rstrip().split('\t')
            c,s,e = fields[0:3]
            if c not in features:
                features[c] = []
            s = int(s)
            e = int(e)
            m = (s+e)/2
            features[c].append([c,s,e,m])
    logging.debug('done')
    return features

def read_chrom_length(filename):
    chrLen = {}
    with open(filename) as fh:
        for line in fh:
            fields = line.rstrip().split('\t')
            c,l = fields[0:2]
            chrLen[c] = int(l)
    logging.debug('done')
    return chrLen

def get_neighbour_boundary(features, chrLen):
    CHROM,START,END,MID = range(4)
    for chrom in features:
        feats = features[chrom]
        n = len(feats)
        for i,f in enumerate(feats):
            if i>0:
                prev_e = feats[i-1][END]
            else:
                prev_e = 0
            if i<n-1:
                next_s = feats[i+1][START]
            else:
                next_s = chrLen[chrom]
            feats[i].extend([prev_e, next_s])
    logging.debug('done')
    return features

def get_center_boundary(features, focal=True):
    CHROM,START,END,MID,PREV_E,NEXT_S = range(6)
    for chrom in features:
        feats = features[chrom]
        n = len(feats)
        for i,f in enumerate(feats):
            if focal:
                feats[i].extend([f[MID],f[MID]])
            else:
                feats[i].extend([f[START],f[END]])
    logging.debug('done')
    return features

def get_flanking(features, w=None, mode='truncated'):
    assert w is not None or mode == 'scaled', '"-f" required when "-m" is not "scaled"'
    CHROM,START,END,MID,PREV_E,NEXT_S,CENT_S,CENT_E = range(8)
    for chrom in features:
        feats = features[chrom]
        for i,f in enumerate(feats):
            s0 = f[CENT_S]
            e0 = f[CENT_E]
            if mode == 'truncated':
                s1 = s0 - w
                if s1 < f[PREV_E]:
                    s1 = f[PREV_E]
                e1 = e0 + w
                if e1 > f[NEXT_S]:
                    e1 = f[NEXT_S]
            elif mode == 'scaled':
                s1 = f[PREV_E]
                e1 = f[NEXT_S]
            elif mode == 'fixed':
                s1 = s0 - w
                e1 = e0 + w
            else:
                raise ValueError('{}: unsupported mode'.format(mode))
            feats[i].extend([s1,e1])
    logging.debug('done')
    return features

def print_features(features):
    for chrom in features:
        feats = features[chrom]
        for f in feats:
            print('\t'.join(map(str, f)))

def print_contact(features1, features2=None, minD=0, maxD=1e8):
    CHROM,START,END,MID,PREV_E,NEXT_S,CENT_S,CENT_E,FLNK_S,FLNK_E = range(10)
    if features2 is None:
        for chrom in features1:
            feats1 = features1[chrom]
            n1 = len(feats1)
            for i in xrange(n1):
                f1 = feats1[i]
                for j in xrange(i+1,n1):
                    f2 = feats1[j]
                    d = f2[MID] - f1[MID]
                    if d > minD and d < maxD:
                        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                            chrom,f1[FLNK_S],f1[FLNK_E],chrom,f2[FLNK_S],f2[FLNK_E],
                            chrom,f1[CENT_S],f1[CENT_E],chrom,f2[CENT_S],f2[CENT_E],i,j,d))
    else:
        for chrom in features1:
            feats1 = features1[chrom]
            feats2 = features2[chrom]
            n1 = len(feats1)
            n2 = len(feats2)
            for i in xrange(n1):
                f1 = feats1[i]
                for j in xrange(n2):
                    f2 = feats2[j]
                    d = abs(f2[MID]-f1[MID])
                    if d > minD and d < maxD:
                        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                            chrom,f1[FLNK_S],f1[FLNK_E],chrom,f2[FLNK_S],f2[FLNK_E],
                            chrom,f1[CENT_S],f1[CENT_E],chrom,f2[CENT_S],f2[CENT_E],i,j,d))

def print_self_contact(features):
    CHROM,START,END,MID,PREV_E,NEXT_S,CENT_S,CENT_E,FLNK_S,FLNK_E = range(10)
    for chrom in features:
        feats = features[chrom]
        for f in feats:
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                chrom,f[FLNK_S],f[FLNK_E],chrom,f[FLNK_S],f[FLNK_E],
                chrom,f[CENT_S],f[CENT_E],chrom,f[CENT_S],f[CENT_E],0,0))


def main(args):
    logging.info(args)
    inputFn1 = args['input1']
    inputFn2 = args['input2']
    flkLen = int(args['f'])
    flkMode = args['m']
    minD = int(args['d'])
    maxD = int(args['D'])

    features1 = read_features(inputFn1)
    chrLen = read_chrom_length(args['g'])

    features1 = get_neighbour_boundary(features1, chrLen)
    if args['center']:
        features1 = get_center_boundary(features1, focal=True)
    else:
        features1 = get_center_boundary(features1, focal=False)
    features1 = get_flanking(features1, w=flkLen, mode=flkMode)

    if inputFn2 is not None:
        features2 = read_features(inputFn2)
        features2 = get_neighbour_boundary(features2, chrLen)
        if args['center']:
            features2 = get_center_boundary(features2, focal=True)
        else:
            features2 = get_center_boundary(features2, focal=False)
        features2 = get_flanking(features2, w=flkLen, mode=flkMode)
    else:
        features2 = None

    if args['domain']:
        print_self_contact(features1)
    else:
        print_contact(features1, features2=features2, minD=minD, maxD=maxD)


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
