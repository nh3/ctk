#!/usr/bin/env python
'''
Usage: cMapCorr.py <input1> <input2>

Options:
    <input1>     input matrix 1
    <input2>     input matrix 2
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import scipy.sparse


def main(args):
    inFn1 = args['input1']
    inFn2 = args['input2']
    mat1 = scipy.sparse.load_npz(inFn1)
    mat2 = scipy.sparse.load_npz(inFn2)


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
