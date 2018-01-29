#!/usr/bin/env python
'''
Usage: cMapBalance.py <inMat> <outMat> <coef>

Options:
    <inMat>     input sparse matrix in scipy npz format
    <outMat>    output balanced matrix in scipy npz format
    <coef>      output normalization coeficients
'''

from __future__ import print_function
import sys
import signal
import logging
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
logging.basicConfig(
        level=logging.WARN,
        format='%(asctime)s; %(levelname)s; %(funcName)s; %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy.sparse

def computeKRNormVector(matrix, tol=1e-6, delta=0.1, Delta=3):
    """
    function [x,res] = bnewt(A,tol,x0,delta,fl)
    % BNEWT A balancing algorithm for symmetric matrices
    %
    % X = BNEWT(A) attempts to find a vector X such that
    % diag(X)*A*diag(X) is close to doubly stochastic. A must
    % be symmetric and nonnegative.
    %
    % X0: initial guess. TOL: error tolerance.
    % delta/Delta: how close balancing vectors can get to the edge of the
    % positive cone. We use a relative measure on the size of elements.
    % FL: intermediate convergence statistics on/off.
    % RES: residual error, measured by norm(diag(x)*A*x - e).
    % Initialise
    [n,n]=size(A); e = ones(n,1); res=[];
    if nargin < 6, fl = 0; end
    if nargin < 5, Delta = 3; end
    if nargin < 4, delta = 0.1; end
    if nargin < 3, x0 = e; end
    if nargin < 2, tol = 1e-6; end
    """
    logging.info('starts')
    n = matrix.shape[0]
    e = np.ones((n,1))
    # inner stopping criterion parameters.
    g = 0.9
    etamax = 0.1
    eta = etamax
    stop_tol = tol*0.5

    x = np.ones((n,1))
    rt = tol**2
    v = x*(matrix*x)
    rk = 1.0 - v
    rho_km1 = np.dot(rk.transpose(), rk)
    rout = rho_km1
    rold = rout

    MVP = 0 # We'll count matrix vector products.
    not_changing = 0

    i = 0 # Outer iteration count.
    while rout > rt and not_changing < 100: # Outer iteration
        i += 1
        k = 0
        y = e
        innertol = max(rt, (eta**2)*rout)
        rho_km2 = rho_km1

        while rho_km1 > innertol: # Inner iteration by CG
            k += 1
            if k == 1:
                Z = rk/v
                p = Z
                rho_km1 = np.dot(rk.transpose(), Z)
            else:
                beta = rho_km1/rho_km2
                p = Z + beta*p
            # Update search direction efficiently
            w = x*(matrix*(x*p)) + v*p
            alpha = rho_km1 / np.dot(p.transpose(), w)
            ap = alpha * p
            # Test distance to boundary of cone
            ynew = y + ap
            if ynew.min() <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0)
                gamma = ((delta-y[ind])/ap[ind]).min()
                y += gamma*ap
                break
            if ynew.max() >= Delta:
                ind = np.where(ynew > Delta)
                gamma = ((Delta-y[ind])/ap[ind]).min()
                y += gamma*ap
                break
            y = ynew
            rk -= alpha*w
            rho_km2 = rho_km1
            Z = rk/v
            rho_km1 = np.dot(rk.transpose(), Z)

        x = x * y
        v = x*(matrix*x)
        rk = 1.0 - v
        rho_km1 = np.dot(rk.transpose(), rk)
        if abs(rho_km1-rout) < tol or np.isinf(rho_km1):
            not_changing += 1
        rout = rho_km1
        MVP += k + 1
        # Update inner iteration stopping criterion.
        rat = rout/rold
        rold = rout
        res_norm = rout**0.5
        eta_o = eta
        eat = g*rat
        if g*(eta_o**2) > 0.1:
            eta = max(eta, g*(eta_o**2))
        eta = max(min(eta, etamax), stop_tol/res_norm)

    logging.info('done')
    if not_changing >= 100:
        return None
    return x


def computeKR(matrix):
    n = matrix.shape[0]
    x0 = np.ones(n) * np.nan
    rs = matrix.sum(axis=1)
    for pct in range(20):
        minD = np.percentile(rs[rs>0], pct)
        k = np.where(rs>=minD)[0]
        x = computeKRNormVector(matrix[k][:,k], tol=1e-6, delta=0.1, Delta=3)
        if x is not None and not np.any(np.isinf(x)):
            x0[k] = x
            break
    return x0


def diagSparseMatrix(x):
    x = x.flatten()
    n = x.shape[0]
    i = np.arange(n)
    return scipy.sparse.csr_matrix((x, (i,i)), shape=(n,n), dtype=x.dtype)


def main(args):
    logging.info(args)
    inputFn = args['inMat']
    outputFn = args['outMat']
    coefFn = args['coef']

    matrix = scipy.sparse.load_npz(inputFn).astype(float)
    x = computeKR(matrix)
    sdx = diagSparseMatrix(x)
    y = sdx * matrix * sdx
    scipy.sparse.save_npz(outputFn, y)
    np.savetxt(coefFn, x, fmt='%.4f', delimiter='\t')
    logging.info('All done')


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.lstrip('-<').rstrip('>'):args[k] for k in args}
    try:
        main(args)
    except KeyboardInterrupt:
        logging.warning('Interrupted')
        sys.exit(1)
