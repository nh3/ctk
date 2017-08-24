#!/usr/bin/env python
'''
Convert SAM to contact matrix

Usage: sam2contact (-c chrLen) (-w width) (-o output) [-s step] [-d minD] [-r region] [-i sam]

Options:
    -c chrLen       chromosome length, genome.fa.fai
    -w width        bin width
    -r region       region, chrom[:start-end]
    -s step         bin step size, default equal to "-w"
    -d minD         minimum mapping distance to be considered informative [default: 1000]
    -i sam          input SAM, read from stdin if omitted
    -o output       output
'''

from __future__ import print_function

import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s; %(levelname)s; %(funcName)s(); %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')
import gzip
import os.path
from os import makedirs
from collections import OrderedDict as ordict
from collections import deque
from math import floor,ceil
import numpy as np
import scipy.sparse

MAX_DENSE = 100000

def readChromLen(filename):
    '''extract an OrderedDict of {chrom:size} from a genome.fa.fai'''
    chrom_size = ordict()
    with open(filename) as f:
        for line in f:
            fields = line.rstrip().split()
            chrom = fields[0]
            size = int(fields[1])
            chrom_size[chrom] = size
    return chrom_size


def iter_sam(sam_fh):
    for line in sam_fh:
        if line[0] == '@':
            continue
        rname,flag,chrom,pos,mapq,cigar,chrom2,pos2,dist,other = line.rstrip().split('\t', 9)
        if chrom2 == '=':
            chrom2 = chrom
        yield rname,flag,chrom,int(pos),chrom2,int(pos2),int(dist)


def iter_bins(bins):
    i = -1
    for chrom in bins:
        for b in bins[chrom]:
            i += 1
            yield i,chrom,b


def parseRegion(region, chrom_size):
    regChr,colon,regCoord = region.partition(':')
    if regChr not in chrom_size:
        raise ValueError("{} is not a valid chromosome".format(regChr))
    if regCoord == '':
        regStart = 0
        regEnd = chrom_size[regChr]
    else:
        s,dash,e = regCoord.partition('-')
        assert dash == '-', "incorrect region format in [{}]".format(region)
        if e == '':
            e = self.chrom_size[regChr]
        if s == '':
            s = 0
        regStart = int(s)
        regEnd = int(e)
        if regEnd > chrom_size[regChr]:
            raise ValueError('region exceeds chromosome size ')
    return regChr,regStart,regEnd


def makeBins(region, chrom_size, width, step=None):
    bins = ordict()
    offsets = ordict()
    nbin = 0
    if step is None:
        step = width
    elif step > width:
        raise ValueError('step size({}) larger than bin width({})'.format(step, width))
    if region is None:
        for chrom in chrom_size:
            offsets[chrom] = nbin
            size = chrom_size[chrom]
            bin_ends = np.arange(width, size+0.1, step).astype(int)
            bin_starts = bin_ends - width
            n = len(bin_ends)
            if bin_ends[n-1] < size:
                bin_ends = np.append(bin_ends, size)
                bin_starts = np.append(bin_starts, bin_starts[n-1]+step)
                n += 1
            bins[chrom] = np.vstack((bin_starts,bin_ends,bin_ends-bin_starts)).transpose()
            nbin += n
    else:
        chrom,start,end = parseRegion(region, chrom_size)
        offsets[chrom] = nbin
        bin_ends = np.arange(start+width, end+0.1, step).astype(int)
        bin_starts = bin_ends - width
        n = len(bin_ends)
        if bin_ends[n-1] < end:
            bin_ends = np.append(bin_ends, end)
            bin_starts = np.append(bin_starts, bin_starts[n-1]+step)
            n += 1
        bins[chrom] = np.vstack((bin_starts,bin_ends,bin_ends-bin_starts)).transpose()
        nbin += n
    logging.info('{} bins made'.format(nbin))
    return bins,offsets,nbin


def getChromBinRange(bins, chrom):
    n,s,e = 0,-1,-1
    if chrom in bins:
        n = len(bins[chrom])
        s = bins[chrom][0][0]
        e = bins[chrom][-1][1]
    return n,s,e


def getPosBinRange(pos, n, s, width, step, offset=0):
    k_min = max(0, int(ceil((pos-s-width+0.1)/step)))
    k_max = min(n-1, int(floor((pos-s)/step)))+1
    return offset+k_min,offset+k_max


def initMatrix(nbin, MAX_DENSE):
    cis_matrix = scipy.sparse.dok_matrix((nbin,nbin), dtype=np.uint16)
    matrixType = 'sparse'
    return cis_matrix,matrixType


def makeMatrix(bins, offsets, nbin, width, step, minD, sam_fh): 
    I = deque()
    J = deque()
    #matrix,matrixType = initMatrix(nbin, MAX_DENSE)
    #nTrans = np.zeros(nbin, dtype=np.uint32)
    #nNonInfo = np.zeros(nbin, dtype=np.uint32)
    #nCisShort = np.zeros(nbin, dtype=np.uint32)
    #nCisLong = np.zeros(nbin, dtype=np.uint32)
    scanned = set()
    for rname,flag,chrom,pos,chrom2,pos2,dist in iter_sam(sam_fh):
        if rname in scanned:
            continue
        n1,s1,e1 = getChromBinRange(bins, chrom)
        if n1 == 0 or pos < s1 or pos > e1:
            continue
        scanned.add(rname)
        k_min1,k_max1 = getPosBinRange(pos, n1, s1, width, step, offsets[chrom])
        n2,s2,e2 = getChromBinRange(bins, chrom2)
        if chrom2 != chrom:
            # trans
            #nTrans[k_min1:k_max1] += 1
            if n2 > 0 and pos >= s1 and pos <= e1:
                # include in matrix
                k_min2,k_max2 = getPosBinRange(pos, n2, s2, width, step, offsets[chrom2])
                #matrix[k_min1:k_max1,k_min2:k_max2] += 1
                #matrix[k_min2:k_max2,k_min1:k_max1] += 1
                I.extend(xrange(k_min1,k_max1))
                I.extend(xrange(k_min2,k_max2))
                J.extend(xrange(k_min2,k_max2))
                J.extend(xrange(k_min1,k_max1))
        else:
            # cis
            if abs(dist) <= minD:
                # non-info, exclude from matrix
                #nNonInfo[k_min1:k_max1] += 1
                pass
            elif pos2 >= s2 and pos2 <= e2:
                # in-range cis info
                k_min2,k_max2 = getPosBinRange(pos2, n2, s2, width, step, offsets[chrom2])
                #matrix[k_min1:k_max1,k_min2:k_max2] += 1
                #matrix[k_min2:k_max2,k_min1:k_max1] += 1
                #nCisShort[k_min1:k_max1] += 1
                I.extend(xrange(k_min1,k_max1))
                I.extend(xrange(k_min2,k_max2))
                J.extend(xrange(k_min2,k_max2))
                J.extend(xrange(k_min1,k_max1))
            else:
                # out-of-range cis info, exclude from matrix
                #nCisLong[k_min1:k_max1] += 1
                pass
    V = np.ones(len(I), dtype=np.uint16)
    matrix = scipy.sparse.coo_matrix((V,(I,J)), shape=(nbin,nbin)).tocsr()
    logging.info('done')
    #return matrixType,matrix,nTrans,nNonInfo,nCisShort,nCisLong
    return matrix


def convertMatrix(dok_matrix):
    return dok_matrix.tocoo().tocsr()


def writeMatrix(matrix, filename):
    scipy.sparse.save_npz(filename, matrix)


#def writeContacts(matrix, bins, fout):
#    for i,chrom1,b1 in iter_bins(bins):
#        bin_mid1 = (b1[0]+b1[1])/2
#        for j,chrom2,b2 in iter_bins(bins):
#            if j<=i:
#                continue
#            contact = matrix[i,j] 
#            if contact > 0:
#                bin_mid2 = (b2[0]+b2[1])/2
#                print('{}\t{}\t{}\t{}\t{}'.format(chrom1,bin_mid1,chrom2,bin_mid2,int(contact)), file=fout)
#    logging.info('done')
#
#
#def writeBins(bins, nTrans, nNonInfo, nCisShort, nCisLong, fout):
#    for i,chrom,b in iter_bins(bins):
#        bin_mid = (b[0]+b[1])/2
#        print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(chrom,b[0],b[1],nTrans[i],nNonInfo[i],nCisShort[i],nCisLong[i]), file=fout)
#    logging.info('done')


def main(args):
    binWidth = int(args['w'])
    minD = int(args['d'])
    if args['s'] is None:
        binStep = binWidth
    else:
        binStep = int(args['s'])
    region = args['r']
    samIn = args['i']
    output = args['o']
    if not output.endswith('.npz'):
        output += '.npz'

    chromLen = readChromLen(args['c'])
    for c in chromLen:
        if chromLen[c] < binWidth:
            del chromLen[c]
    bins,offsets,nbin = makeBins(region, chromLen, binWidth, binStep)

    if samIn is None:
        sam_fh = sys.stdin
    else:
        sam_fh = open(samIn)
    #matrixType,matrix,nTrans,nNonInfo,nCisShort,nCisLong = makeMatrix(bins, offsets, nbin, binWidth, binStep, minD, sam_fh)
    matrix = makeMatrix(bins, offsets, nbin, binWidth, binStep, minD, sam_fh)
    sam_fh.close()

    #writeMatrix(convertMatrix(matrix), output)
    writeMatrix(matrix, output)

    #outdir = os.path.dirname(prefix)
    #if outdir and not os.path.isdir(outdir):
    #    makedirs(outdir)
    #bins_fh = gzip.open(prefix + 'bins.txt.gz', 'wb', 5)
    #writeBins(bins, nTrans, nNonInfo, nCisShort, nCisLong, bins_fh)
    #bins_fh.close()

    #contacts_fh = gzip.open(prefix + 'contacts.txt.gz', 'wb', 5)
    #writeContacts(matrix, bins, contacts_fh)
    #contacts_fh.close()

    #np.savetxt(prefix + 'matrix.txt.gz', matrix, fmt='%.4f', delimiter='\t')

    logging.info('All done successfully')



if __name__ == '__main__':
    import cProfile
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.lstrip('-<').rstrip('>'):args[k] for k in args}
    try:
        main(args)
        #cProfile.run('main(args)')
    except KeyboardInterrupt:
        logging.warning('Interrupted')
        sys.exit(1)
