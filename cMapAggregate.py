#!/usr/bin/env python
"""
Aggregate contact analysis

Usage: cMapAggregate.py [options] <genome> <cmap> <contact> [<random_contact>]

Options:
    -m <mode>           {loop:
                           "-x 1 -y 10 -g fc_bg -M perm --fixed --offdiag",
                         domain:
                           "-x 9 -y 5 -g fc_side -M simple",
                         compartment:
                           "-x 9 -y 5 -g fc_side -M simple --offdiag"}.
                        Can be overridden by individual options.
    -x <n_cntr>         An odd number of central bins.
    -y <n_flnk>         Number of flanking bins.
    --fixed             Fixed sub-mats dimensions, otherwise central and
                        flanking areas are scaled to sizes given by -x and -y.
    --offdiag           Require sub matrix to not touch the diagonal.
    -M <method>         Method for calculating p-values.
                        {perm: permutation-based,
                         simple: single t-test, implies -N 0}
    -N <n_perm>         Number of permutations, required for -M perm.
                        [default: 1000]
    -g <bg>             Background type, {fc_bg, fc_side, fc_diag}
    -r <res>            Resolution in bp or human readable format.
                        [default: 1k]
    -t <sparsity>       Only submats with sparsities greater than this value
                        are kept. [default: 0.0]
    -F <fc_filter>      Remove outlier submats based on fc, in the format of
                        "<method>:<n_sigma>", where methods are
                        {sd: "std", mad: "mad"} [default: "mad:5"]
    -b <ddbg>           Prefix of distance-decay background tables. For cMap
                        in npz format, ddbg tables for individual chromosome
                        will be <ddbg>.<chr>.KR.ddbg, which have one-column
                        background contact frequency per line/bin.
    -s <seed>           Integer used as random seed for generating
                        permutations. [default: 0]
    -f <cmap_fmt>       cMap format, determines cMap suffix.
                        {h5: ".h5", npz: ".<chr>.KR.npz"} [default: npz]
    -o <out_prfx>       Output prefix. Outputs will be
                        image: <out_prfx>.c<n_cntr>_f<n_flnk>.apa.pdf
                        matrix: <out_prfx>.c<n_cntr>_f<n_flnk>.apa.mat
                        stats: <out_prfx>.c<n_cntr>_f<n_flnk>.apa.stat
                        table: <out_prfx>.c<n_cntr>_f<n_flnk>.apa.txt
                        [default: output]
    --plot-opt <opt>    Plot options in "key1:value1,key2:value2" format.
                        Available keys are: do_log(True), fc_type(fc_bg),
                        max_lfc(2), add_label(True), version(''). [default: '']
    --plot-only         plot only, require data generated already
    --quiet             Print only warning and error messages
    --debug             Print debug information
    --prof              Print profile informaion

Arguments:
    <genome>            A two column table listing chromosome names and lengths
    <cmap>              Prefix of cmaps. For npz format, cmap for individual
                        chromosome will be <cmap>.<chr>.<suffix>.
    <contact>           A 15-column contact table, the last three column
                        being <dist-band>, <cov-band1>, <cov-band2>
    <random_contact>    Suffix of random contact tables, same format as
                        <contact>, where tables for individual chromosome will
                        be <random_contact>.<chr>.txt.gz
"""

from __future__ import print_function
from builtins import range
import sys
import signal
import logging
import os
from collections import Counter
import re
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.stats import ttest_ind
from scipy.ndimage.filters import gaussian_filter
from statsmodels import robust
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

plt.switch_backend('Agg')
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def memodict(f):
    """ Memoization decorator for a function taking a single argument """

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__


def parse_resolution(res):
    unit = res[-1]
    if unit.lower() == 'k':
        res = int(res[0:-1]) * 1000
    elif unit.lower() == 'm':
        res = int(res[0:-1]) * 1000000
    elif unit in '0123456789':
        res = int(res)
    return res


def parse_filter_opts(opt_str):
    filter_opts = {}
    method, sep, n_sigma = opt_str.strip("'\"").rstrip("'\"").partition(':')
    if sep == ':':
        if method not in ['sd', 'mad']:
            raise NotImplementedError(method)
        filter_opts['method'] = method
        filter_opts['n_sigma'] = int(n_sigma)
    return filter_opts


def parse_plot_opts(opt_str):
    plot_opts = {}
    for opt in opt_str.split(','):
        k, sep, v = opt.partition(':')
        if sep != ':':
            continue
        if re.match(r'[0-9]+$', v):
            v = int(v)
        elif re.match(r'[0-9]+[.][0-9]*|[0-9]+([.][0-9])?[eE][+-]?[0-9]+$', v):
            v = float(v)
        elif v.lower() == 'false':
            v = False
        elif v.lower() == 'none':
            v = None
        plot_opts[k] = v
    return plot_opts


def read_genome(fname):
    chroms = pd.read_csv(
        fname,
        sep='\t',
        header=None,
        comment='#',
        names=['chrom', 'length'],
        dtype={'chrom': str, 'length': int},
    )
    logging.debug('done')
    return chroms


def read_cmap(fn_prfx, chrom, fmt='npz', balanced=True):
    assert fmt in ('npz', 'h5'), 'unsupported cMap format: {}'.format(fmt)
    if fmt == 'npz':
        if balanced:
            suffix = 'KR.npz'
        else:
            suffix = 'npz'
        fname = '{}.{}.{}'.format(fn_prfx, chrom, suffix)
        cmap = np.nan_to_num(load_npz(fname).toarray())
        n = cmap.shape[0]
        idx0 = np.arange(n)
        idx1 = np.arange(n - 1)
        x1 = cmap[idx1, idx1 + 1]
        x0 = (np.insert(x1, 0, 0) + np.append(x1, 0)) / 2
        cmap[idx0, idx0] = x0
    else:
        raise NotImplementedError
    logging.debug('done')
    return cmap


def read_ddbg(fn_prfx, chrom, cmap_fmt='npz', balanced=True):
    if fn_prfx is None:
        return None
    assert cmap_fmt in ('npz', 'h5'), 'unsupported cMap format: {}'.format(cmap_fmt)
    if cmap_fmt == 'npz':
        if balanced:
            suffix = 'KR.ddbg'
        else:
            suffix = 'ddbg'
        fname = '{}.{}.{}'.format(fn_prfx, chrom, suffix)
        tbl = pd.read_csv(fname, sep='\t', header=None, names=['ddbg'], dtype={'ddbg': float})
    else:
        raise NotImplementedError
    logging.debug('done')
    return tbl.ddbg.values


def read_contacts(fname):
    contacts = pd.read_csv(
        fname,
        sep='\t',
        header=None,
        names=[
            'c1',
            's1',
            'e1',
            'c2',
            's2',
            'e2',
            'c1c',
            's1c',
            'e1c',
            'c2c',
            's2c',
            'e2c',
            'dband',
            'cband1',
            'cband2',
        ],
        dtype={
            'c1': str,
            's1': int,
            'e1': int,
            'c2': str,
            's2': int,
            'e2': int,
            'c1c': str,
            's1c': int,
            'e1c': int,
            'c2c': str,
            's2c': int,
            'e2c': int,
            'dband': float,
            'cband1': float,
            'cband2': float,
        },
    )
    contacts = contacts.dropna()
    contacts.dband = np.round(contacts.dband.values).astype(int)
    contacts.cband1 = np.round(contacts.cband1.values).astype(int)
    contacts.cband2 = np.round(contacts.cband2.values).astype(int)
    logging.debug('done. {} returned'.format(len(contacts)))
    return contacts


def map_contacts_to_cmap(contacts, res):
    contacts['x1'] = np.round(contacts.s1 / res).astype(int)
    contacts['x2'] = np.round(contacts.e1 / res).astype(int)
    contacts['x1c'] = np.round(contacts.s1c / res).astype(int)
    contacts['x2c'] = np.round(contacts.e1c / res).astype(int)
    contacts['y1'] = np.round(contacts.s2 / res).astype(int)
    contacts['y2'] = np.round(contacts.e2 / res).astype(int)
    contacts['y1c'] = np.round(contacts.s2c / res).astype(int)
    contacts['y2c'] = np.round(contacts.e2c / res).astype(int)
    logging.debug('done')
    return contacts


def filter_contacts_by_shape(
    contacts,
    cmap_size,
    n_cntr,
    n_flnk,
    min_n_cntr=1,
    min_n_flnk=2,
    fixed_shape=False,
    offdiag=False,
):
    logging.debug('pre-filter: {}'.format(len(contacts)))
    k1 = (contacts.x1 >= 0) & (contacts.x2 < cmap_size)
    k2 = (contacts.y1 >= 0) & (contacts.y2 < cmap_size)
    if fixed_shape:
        k3 = ((contacts.x2 - contacts.x1) == n_flnk * 2 + n_cntr) & (
            (contacts.x2c - contacts.x1c) == n_cntr
        )
        k4 = ((contacts.y2 - contacts.y1) == n_flnk * 2 + n_cntr) & (
            (contacts.y2c - contacts.y1c) == n_cntr
        )
    else:
        k3 = (
            ((contacts.x1 + min_n_flnk) <= contacts.x1c)
            & ((contacts.x1c + min_n_cntr) <= contacts.x2c)
            & ((contacts.x2c + min_n_flnk) <= contacts.x2)
        )
        k4 = (
            ((contacts.y1 + min_n_flnk) <= contacts.y1c)
            & ((contacts.y1c + min_n_cntr) <= contacts.y2c)
            & ((contacts.y2c + min_n_flnk) <= contacts.y2)
        )
    if offdiag:
        k5 = contacts.x2 + 1 < contacts.y1
        f_contacts = contacts[k1 & k2 & k3 & k4 & k5]
    else:
        f_contacts = contacts[k1 & k2 & k3 & k4]
    logging.debug('done. {} returned'.format(len(f_contacts)))
    return f_contacts


def calc_contact_band_stat(contacts):
    k = contacts.cband1 > contacts.cband2
    contacts['bands'] = ''
    contacts.loc[k, 'bands'] = (
        contacts.loc[k, 'dband'].map(str)
        + ','
        + contacts.loc[k, 'cband2'].map(str)
        + ','
        + contacts.loc[k, 'cband1'].map(str)
    )
    contacts.loc[~k, 'bands'] = (
        contacts.loc[~k, 'dband'].map(str)
        + ','
        + contacts.loc[~k, 'cband1'].map(str)
        + ','
        + contacts.loc[~k, 'cband2'].map(str)
    )
    return Counter(contacts.bands)


def generate_matched_random_contacts(contacts, random_contacts, n_set=1000):
    bdcnt = calc_contact_band_stat(contacts)
    rnd_bdcnt = calc_contact_band_stat(random_contacts)
    idx_by_group = random_contacts.index.groupby(random_contacts.bands)
    matched_random_sets = []
    for _ in range(n_set):
        idx = []
        for bd, cnt in bdcnt.items():
            rnd_cnt = rnd_bdcnt[bd]
            ridx = np.random.randint(0, rnd_cnt, cnt)
            idx.extend(idx_by_group[bd][ridx])
        matched_random_sets.append(sorted(idx))
    return matched_random_sets


@memodict
def make_asymmetric_distance_matrix(dim):
    h, w = map(int, dim.split(','))
    if h >= w:
        n = h
        i = (h - w) / 2
    else:
        n = w
        i = (w - h) / 2
    d = np.arange(n, dtype=int)
    dmat = np.repeat(d.reshape(1, n), n, axis=0) - d.reshape(n, 1)
    if h >= w:
        return dmat[:, i: (i + w)]
    else:
        return dmat[i: (i + h), :]


@memodict
def make_symmetric_distance_matrix(dim):
    h, w = map(int, dim.split(','))
    if h >= w:
        n = h
        i = (h - w) / 2
    else:
        n = w
        i = (w - h) / 2
    d = np.arange(n, dtype=int)
    dmat = np.abs(np.repeat(d.reshape(1, n), n, axis=0) - d.reshape(n, 1))
    if h >= w:
        return dmat[:, i: (i + w)]
    else:
        return dmat[i: (i + h), :]


def extract_sub_matrix(cmap, contacts, ddbg=None, make_distance_matrix=None):
    if ddbg is not None:
        submats = [
            cmap[i1:i2, j1:j2]
            / ddbg[
                make_distance_matrix('{},{}'.format(i2 - i1, j2 - j1))
                + (j1 + j2) // 2
                - (i1 + i2) // 2
            ]
            for i1, i2, j1, j2 in contacts[['x1', 'x2', 'y1', 'y2']].values
        ]
    else:
        submats = [
            cmap[i1:i2, j1:j2]
            for i1, i2, j1, j2 in contacts[['x1', 'x2', 'y1', 'y2']].values
        ]
    return submats


def filter_submats_by_sparsity(submats, min_sparsity=0.0):
    sm_sparsity = np.array(
        [(sm > 0).sum().astype(float) / (sm.shape[0] * sm.shape[1]) for sm in submats]
    )
    k = sm_sparsity > min_sparsity
    logging.debug(
        '{} submats passed sparsity threshold {}'.format(k.sum(), min_sparsity)
    )
    return k


@memodict
def _scale_idx(shape):
    n, N = shape
    nx = np.arange(n)
    Nx = np.linspace(0, n - 1, N)
    return nx, Nx


def _scale_mat(mat, M, N):
    m, n = mat.shape
    tgt = None
    if m == M and n == N:
        return mat
    elif m > 1 and n > 1:
        mx, Mx = _scale_idx((m, M))
        nx, Nx = _scale_idx((n, N))
        km = min(3, m - 1)
        kn = min(3, n - 1)
        rbs = RectBivariateSpline(mx, nx, mat, kx=km, ky=kn)
        tgt = rbs(Mx, Nx)
    elif m == 1 and n == 1:
        tgt = np.full((M, N), mat[0, 0])
    elif m == 1:
        nx, Nx = _scale_idx((n, N))
        kn = min(3, n - 1)
        spl = UnivariateSpline(nx, mat[0], k=kn)
        tgt = np.repeat(spl(Nx).reshape((1, N)), M, axis=0)
    elif n == 1:
        mx, Mx = _scale_idx((m, M))
        km = min(3, m - 1)
        spl = UnivariateSpline(mx, mat[:, 0], k=km)
        tgt = np.repeat(spl(Mx).reshape((M, 1)), N, axis=1)
    else:
        logging.error(mat)
        logging.error((n, m))
        logging.error('should not reach here')
    return tgt


def scale_submat(mat, contact, n_cntr, n_flnk):
    X, Y = mat.shape
    N = n_cntr + 2 * n_flnk
    if X == N and Y == N:
        return mat
    z = n_cntr + n_flnk
    new_mat = np.zeros((N, N))

    x1 = int(contact.x1c - contact.x1)
    x2 = int(contact.x2c - contact.x1)
    y1 = int(contact.y1c - contact.y1)
    y2 = int(contact.y2c - contact.y1)

    top = mat[0:x1, y1:y2]
    left = mat[x1:x2, 0:y1]
    bottom = mat[x2:X, y1:y2]
    right = mat[x1:x2, y2:Y]
    center = mat[x1:x2, y1:y2]
    topleft = mat[0:x1, 0:y1]
    topright = mat[0:x1, y2:Y]
    bottomleft = mat[x2:X, 0:y1]
    bottomright = mat[x2:X, y2:Y]

    new_mat[0:n_flnk, 0:n_flnk] = _scale_mat(topleft, n_flnk, n_flnk)
    new_mat[0:n_flnk, n_flnk:z] = _scale_mat(top, n_flnk, n_cntr)
    new_mat[0:n_flnk, z:N] = _scale_mat(topright, n_flnk, n_flnk)
    new_mat[n_flnk:z, 0:n_flnk] = _scale_mat(left, n_cntr, n_flnk)
    new_mat[n_flnk:z, n_flnk:z] = _scale_mat(center, n_cntr, n_cntr)
    new_mat[n_flnk:z, z:N] = _scale_mat(right, n_cntr, n_flnk)
    new_mat[z:N, 0:n_flnk] = _scale_mat(bottomleft, n_flnk, n_flnk)
    new_mat[z:N, n_flnk:z] = _scale_mat(bottom, n_flnk, n_cntr)
    new_mat[z:N, z:N] = _scale_mat(bottomright, n_flnk, n_flnk)
    return new_mat


def aggregate_contacts(
    cmap,
    contacts,
    n_cntr,
    n_flnk,
    ddbg=None,
    min_sparsity=0.0,
    fixed_shape=False,
    symmetric=False,
):
    if symmetric:
        make_distance_matrix = make_symmetric_distance_matrix
    else:
        make_distance_matrix = make_asymmetric_distance_matrix
    submats = extract_sub_matrix(
        cmap, contacts, ddbg=ddbg, make_distance_matrix=make_distance_matrix
    )
    k = filter_submats_by_sparsity(submats, min_sparsity=min_sparsity)
    submats = [submats[i] for i in np.where(k)[0]]
    contacts = contacts.loc[k]
    if len(submats) > 0:
        if not fixed_shape:
            submats = np.array(
                [
                    scale_submat(submats[i], contacts.iloc[i], n_cntr, n_flnk)
                    for i in range(len(submats))
                ]
            )
        else:
            submats = np.array(submats)
    else:
        n = n_cntr + 2 * n_flnk
        submats = np.zeros((0, n, n))
    logging.info('{} submats'.format(len(submats)))
    logging.debug('done')
    return contacts, submats


def squash_submats(mats, method='mean'):
    if method == 'mean':
        squash = np.nanmean
    squashed = squash(mats, axis=0)
    x = np.nanmedian(squashed)
    if np.isnan(x):
        x = 0
    squashed[np.isnan(squashed)] = x
    squashed[np.isinf(squashed)] = x
    return squashed


def calc_squashed_stats(mat, n_cntr, n_flnk):
    M, N = mat.shape
    x1 = n_flnk
    x2 = n_flnk + n_cntr

    centre = mat[x1:x2, x1:x2]
    top = mat[0:x1, x1:x2]
    left = mat[x1:x2, 0:x1]
    bottom = mat[x2:M, x1:x2]
    right = mat[x1:x2, x2:N]
    topleft = mat[0:x1, 0:x1]
    topright = mat[0:x1, x2:N]
    bottomleft = mat[x2:M, 0:x1]
    bottomright = mat[x2:M, x2:N]

    mat_sum = np.nansum(mat)
    centre_sum = np.nansum(centre)
    bg_sum = mat_sum - centre_sum
    tlbr_sum = np.nansum(top) + np.nansum(left) + np.nansum(bottom) + np.nansum(right)
    diag_sum = np.nansum(topleft) + np.nansum(bottomright)
    offdiag_sum = np.nansum(bottomleft) + np.nansum(topright)

    centre_mean = centre_sum / n_cntr / n_cntr
    bg_mean = bg_sum / (M * N - 1)
    tlbr_mean = tlbr_sum / (n_cntr * n_flnk * 4)
    diag_mean = diag_sum / (n_flnk * n_flnk * 2)
    offdiag_mean = offdiag_sum / (n_flnk * n_flnk * 2)

    fc_bg = centre_mean / bg_mean
    fc_side = centre_mean / tlbr_mean
    fc_diag = diag_mean / offdiag_mean
    return centre_mean, fc_bg, fc_side, fc_diag


def calc_submat_stats(mats, n_cntr, n_flnk, save_data=True):
    L, M, N = mats.shape
    stats = {
        'centres': [],
        'bgs': [],
        'sides': [],
        'diags': [],
        'offdiags': [],
        'n': L,
        'mats': mats,
    }
    if save_data and L > 0:
        x1 = n_flnk
        x2 = n_flnk + n_cntr
        sums = np.nansum(np.nansum(mats, axis=1), axis=1)
        if n_cntr == 1:
            centres = mats[:, x1, x1]
            hstripe_sums = np.nansum(mats[:, x1, :], axis=1)
            vstripe_sums = np.nansum(mats[:, :, x1], axis=1)
            centre_means = centres
            bg_means = (sums - centres) / (M * N - 1)
            tlbr_means = (hstripe_sums + vstripe_sums - 2 * centres) / (M + N - 2)
        else:
            centres = mats[:, x1:x2, x1:x2]
            tops = mats[:, 0:x1, x1:x2]
            lefts = mats[:, x1:x2, 0:x1]
            bottoms = mats[:, x2:M, x1:x2]
            rights = mats[:, x1:x2, x2:N]
            centre_sums = np.nansum(np.nansum(centres, axis=1), axis=1)
            top_sums = np.nansum(np.nansum(tops, axis=1), axis=1)
            left_sums = np.nansum(np.nansum(lefts, axis=1), axis=1)
            bottom_sums = np.nansum(np.nansum(bottoms, axis=1), axis=1)
            right_sums = np.nansum(np.nansum(rights, axis=1), axis=1)
            centre_means = centre_sums / n_cntr / n_cntr
            bg_means = (sums - centre_sums) / (M * N - n_cntr * n_cntr)
            tlbr_means = (top_sums + left_sums + bottom_sums + right_sums) / (
                n_cntr * n_flnk * 4
            )

        toplefts = mats[:, 0:x1, 0:x1]
        toprights = mats[:, 0:x1, x2:N]
        bottomlefts = mats[:, x2:M, 0:x1]
        bottomrights = mats[:, x2:M, x2:N]
        topleft_sums = np.nansum(np.nansum(toplefts, axis=1), axis=1)
        topright_sums = np.nansum(np.nansum(toprights, axis=1), axis=1)
        bottomleft_sums = np.nansum(np.nansum(bottomlefts, axis=1), axis=1)
        bottomright_sums = np.nansum(np.nansum(bottomrights, axis=1), axis=1)
        diag_means = (topleft_sums + bottomright_sums) / (n_flnk * n_flnk * 2)
        offdiag_means = (bottomleft_sums + topright_sums) / (n_flnk * n_flnk * 2)

        stats['centres'] = centre_means
        stats['bgs'] = bg_means
        stats['sides'] = tlbr_means
        stats['diags'] = diag_means
        stats['offdiags'] = offdiag_means
    return stats


def filter_submats_by_stats(stats, fc_type='fc_bg', method='mad', n_sigma=5):
    if method == 'sd':
        estimate_mu = np.nanmean
        estimate_sigma = np.nanstd
    elif method == 'mad':
        estimate_mu = np.nanmedian
        estimate_sigma = robust.mad
    else:
        raise NotImplementedError
    if len(stats['centres']):
        if fc_type == 'fc_bg':
            signal = stats['centres']
            noise = stats['bgs']
        elif fc_type == 'fc_side':
            signal = stats['centres']
            noise = stats['sides']
        elif fc_type == 'fc_bg':
            signal = stats['diags']
            noise = stats['offdiags']
        else:
            raise NotImplementedError
        noise = np.where(noise > 0, noise, stats['bgs'])
        # signal = np.where(noise > 0, signal, signal + np.nanmin(signal[signal > 0]))
        # noise = np.where(noise > 0, noise, noise + np.nanmin(noise[noise > 0]))
        fc = signal / noise
        mu = estimate_mu(fc)
        sigma = estimate_sigma(fc[~np.isnan(fc)])
        logging.info('mu = {}, sigma = {}'.format(mu, sigma))
        k = np.logical_and((fc >= 0), (fc < 40))
        stats['centres'] = stats['centres'][k]
        stats['bgs'] = stats['bgs'][k]
        stats['sides'] = stats['sides'][k]
        stats['diags'] = stats['diags'][k]
        stats['offdiags'] = stats['offdiags'][k]
        stats['n'] = sum(k)
        stats['mat'] = squash_submats(stats['mats'][k])
    else:
        stats['mat'] = squash_submats(stats['mats'])
    del stats['mats']
    logging.debug('done. {} returned'.format(stats['n']))
    return stats


def calc_permutation_p(stats, perm_stats):
    N = float(len(perm_stats))
    perm_fc_bgs = np.array([st['summary']['fc_bg'] for st in perm_stats])
    p_bg = sum(perm_fc_bgs > stats['summary']['fc_bg']) / N
    perm_fc_sides = np.array([st['summary']['fc_side'] for st in perm_stats])
    p_side = sum(perm_fc_sides > stats['summary']['fc_side']) / N
    perm_fc_diags = np.array([st['summary']['fc_diag'] for st in perm_stats])
    p_diag = sum(perm_fc_diags > stats['summary']['fc_diag']) / N
    return {'p_bg': p_bg, 'p_side': p_side, 'p_diag': p_diag}


def calc_simple_p(stats):
    p_bg = ttest_ind(stats['data']['centres'], stats['data']['bgs'], equal_var=False)[1]
    p_side = ttest_ind(
        stats['data']['centres'], stats['data']['sides'], equal_var=False
    )[1]
    p_diag = ttest_ind(
        stats['data']['diags'], stats['data']['offdiags'], equal_var=False
    )[1]
    return {'p_bg': p_bg, 'p_side': p_side, 'p_diag': p_diag}


def prepare_outputs(out_prfx, n_cntr, n_flnk):
    out_dir, prfx = os.path.split(out_prfx)
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif not os.path.isdir(out_dir):
            logging.error(
                'output dir {} already exists' 'and is not a folder'.format(out_dir)
            )
            sys.exit(1)
    out_prfx = '{}.c{}_f{}.apa'.format(out_prfx.rstrip('.'), n_cntr, n_flnk)
    pdf_fn = out_prfx + '.pdf'
    mat_fn = out_prfx + '.mat'
    stat_fn = out_prfx + '.stat'
    tbl_fn = out_prfx + '.txt.gz'
    return {'pdf': pdf_fn, 'mat': mat_fn, 'stat': stat_fn, 'tbl': tbl_fn}


def prepare_permutation_outputs(out_prfx, n_cntr, n_flnk):
    prfx = '{}.c{}_f{}.random_apa'.format(out_prfx.rstrip('.'), n_cntr, n_flnk)
    mat_fn = prfx + '.mat.npy'
    stat_fn = prfx + '.stat'
    return {'mat': mat_fn, 'stat': stat_fn}


def combine_chrom_stats(stats, n_cntr, n_flnk, n_perm=0):
    if n_perm == 0:
        gw_stats = {
            'mat': None,
            'summary': {'n': 0},
            'data': {
                'centres': [],
                'bgs': [],
                'sides': [],
                'diags': [],
                'offdiags': [],
            },
        }
        for chrom in stats:
            stat = stats[chrom]
            if stat['n'] == 0:
                continue
            gw_stats['data']['centres'].extend(stat['centres'])
            gw_stats['data']['bgs'].extend(stat['bgs'])
            gw_stats['data']['sides'].extend(stat['sides'])
            gw_stats['data']['diags'].extend(stat['diags'])
            gw_stats['data']['offdiags'].extend(stat['offdiags'])
            gw_stats['summary']['n'] += stat['n']
            if gw_stats['mat'] is None:
                gw_stats['mat'] = stat['mat'] * stat['n']
            else:
                gw_stats['mat'] += stat['mat'] * stat['n']
        gw_stats['mat'] /= gw_stats['summary']['n']
        centre, fc_bg, fc_side, fc_diag = calc_squashed_stats(
            gw_stats['mat'], n_cntr, n_flnk
        )
        gw_stats['summary']['centre'] = centre
        gw_stats['summary']['fc_bg'] = fc_bg
        gw_stats['summary']['fc_side'] = fc_side
        gw_stats['summary']['fc_diag'] = fc_diag
    else:
        gw_stats = []
        for i in range(n_perm):
            stats_i = {chrom: stats[chrom][i] for chrom in stats}
            gw_stats.append(combine_chrom_stats(stats_i, n_cntr, n_flnk, n_perm=0))
    return gw_stats


def heatmap(ax, x, xmin, xmax, cm, label=None, log=False):
    n = x.shape[0]
    m = n // 2
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    if log:
        im = ax.imshow(np.log2(x), cmap=cm, vmin=np.log2(xmin), vmax=np.log2(xmax))
    else:
        im = ax.imshow(x, cmap=cm, vmin=xmin, vmax=xmax)
    plt.colorbar(im)
    if label:
        ax.annotate(
            label,
            xy=(m, m),
            xytext=(m + 2, m),
            verticalalignment='center',
            arrowprops=dict(
                width=0, headlength=10, headwidth=3, facecolor='black', shrink=0.05
            ),
        )


def plot_aggregated_image(
    pdf_fn, stats, max_lfc=2, do_log=True, fc_type='fc_bg', add_label=True, version=None
):
    mat = stats['mat']
    bg = stats['summary']['centre'] / stats['summary'][fc_type]
    M, N = mat.shape
    if do_log:
        cm = LinearSegmentedColormap.from_list(
            'BlWhRd', [(0, 'blue'), (0.5, 'white'), (1, 'red')]
        )
    else:
        cm = LinearSegmentedColormap.from_list('WhRd', [(0, 'white'), (1, 'red')])
    if add_label:
        label = np.round(stats['summary'][fc_type], 2)
    else:
        label = None

    if version:
        pdf_fn = pdf_fn[:-3] + version + '.pdf'
    pp = PdfPages(pdf_fn)
    ax = plt.subplot()
    heatmap(ax, mat / bg, 2 ** (-max_lfc), 2 ** (max_lfc), cm, log=do_log, label=label)
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    plt.close()

    sm = gaussian_filter(mat / bg, 1, mode='nearest')
    ax = plt.subplot()
    heatmap(ax, sm, 2 ** (-max_lfc), 2 ** (max_lfc), cm, log=do_log, label=label)
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    plt.close()
    pp.close()


def write_outputs(outputs, stats, p_values={}, **plot_opts):
    if isinstance(stats, dict):
        np.savetxt(outputs['mat'], stats['mat'])
        name = os.path.basename(outputs['stat']).replace('.apa.stat', '')
        stats['summary']['name'] = name
        for k, v in p_values.items():
            stats['summary'][k] = v
        pd.DataFrame(stats['summary'], index=[0]).to_csv(
            outputs['stat'], sep='\t', index=None
        )
        pd.DataFrame(stats['data']).to_csv(
            outputs['tbl'], sep='\t', index=None, compression='gzip'
        )
        plot_aggregated_image(outputs['pdf'], stats, **plot_opts)
    elif isinstance(stats, list):
        mats = np.array([st['mat'] for st in stats])
        np.save(outputs['mat'], mats, allow_pickle=False)
        summary = {k: [st['summary'][k] for st in stats] for k in stats[0]['summary']}
        pd.DataFrame(summary).to_csv(outputs['stat'], sep='\t', index=None)


def read_outputs(outputs, mat=True, summary=True, data=True):
    stats = {}
    if mat:
        if outputs['mat'].endswith('.npy'):
            stats['mat'] = np.load(outputs['mat'])
        else:
            stats['mat'] = np.loadtxt(outputs['mat'])
    if summary:
        stats['summary'] = pd.read_csv(outputs['stat'], sep='\t').iloc[0].to_dict()
    if data:
        stats['data'] = pd.read_csv(outputs['tbl'], sep='\t').to_dict(orient='list')
    return stats


def main(args):
    logging.debug(args)
    res = parse_resolution(args['r'])
    min_sparsity = float(args['t'])
    cmap_fmt = args['f']
    seed = int(args['s'])

    mode = args['m']
    if mode == 'loop':
        n_cntr = 1
        n_flnk = 10
        method = 'perm'
        fc_type = 'fc_bg'
        fixed_shape = True
        offdiag = True
        symmetric = False
        plot_log = True
    elif mode == 'domain':
        n_cntr = 9
        n_flnk = 5
        method = 'simple'
        fc_type = 'fc_side'
        fixed_shape = False
        offdiag = False
        symmetric = True
        plot_log = True
    elif mode == 'compartment':
        n_cntr = 9
        n_flnk = 5
        method = 'simple'
        fc_type = 'fc_side'
        fixed_shape = False
        offdiag = True
        symmetric = False
        plot_log = True
    elif mode is not None:
        logging.error('unsupported mode')
        sys.exit(1)
    else:
        n_cntr = 1
        n_flnk = 10
        method = 'simple'
        fc_type = 'fc_bg'
        fixed_shape = args['fixed']
        offdiag = args['offdiag']
        symmetric = False
        plot_log = True

    if args['x'] is not None:
        n_cntr = int(args['x'])
    assert (n_cntr // 2) * 2 + 1 == n_cntr, '-x needs to be an odd number'
    if args['y'] is not None:
        n_flnk = int(args['y'])
    if args['M'] is not None:
        method = args['M']
    if args['g'] is not None:
        fc_type = args['g']
    assert fc_type in ('fc_bg', 'fc_side', 'fc_diag'), 'unsupported label type'

    plot_opts = parse_plot_opts(args['plot-opt'])
    plot_opts['fc_type'] = fc_type
    plot_opts['do_log'] = plot_log
    logging.debug(plot_opts)

    filter_opts = parse_filter_opts(args['F'])

    n_perm = int(args['N'])
    ddbg_prfx = args['b']
    out_prfx = args['o']
    genome_fn = args['genome']
    cmap_prfx = args['cmap']
    contact_fn = args['contact']
    random_contact_prfx = args['random_contact']
    assert (
        method == 'simple' or random_contact_prfx is not None
    ), '<random_contact> required for --method perm'
    if method == 'simple':
        n_perm = 0

    outputs = prepare_outputs(out_prfx, n_cntr, n_flnk)

    genome = read_genome(genome_fn)

    if args['plot-only']:
        stats = read_outputs(outputs, mat=True, summary=True, data=False)
        plot_aggregated_image(outputs['pdf'], stats, **plot_opts)
        return 0

    all_contacts = read_contacts(contact_fn)
    obsvd_stats = {}
    permutation_stats = {}
    for chrom in genome.chrom.values:
        logging.info('Working on {}'.format(chrom))
        contacts = all_contacts.loc[all_contacts.c1 == chrom].copy()
        if contacts.empty:
            logging.warning('no contacts found on {}'.format(chrom))
            continue
        cmap = read_cmap(cmap_prfx, chrom, fmt=cmap_fmt)
        cmap_size = cmap.shape[0]
        ddbg = read_ddbg(ddbg_prfx, chrom, cmap_fmt=cmap_fmt)
        contacts = map_contacts_to_cmap(contacts, res)
        contacts = filter_contacts_by_shape(
            contacts,
            cmap_size,
            n_cntr,
            n_flnk,
            fixed_shape=fixed_shape,
            offdiag=offdiag,
        )
        contacts, submats = aggregate_contacts(
            cmap,
            contacts,
            n_cntr,
            n_flnk,
            ddbg=ddbg,
            min_sparsity=min_sparsity,
            fixed_shape=fixed_shape,
            symmetric=symmetric,
        )
        obsvd_stats[chrom] = filter_submats_by_stats(
            calc_submat_stats(submats, n_cntr, n_flnk),
            fc_type=fc_type,
            method=filter_opts['method'],
            n_sigma=filter_opts['n_sigma'],
        )
        if method == 'perm':
            permutation_stats[chrom] = []
            np.random.seed(seed)
            rand_contact_fn = '{}.{}.txt.gz'.format(random_contact_prfx, chrom)
            rand_contact_pool = read_contacts(rand_contact_fn)
            rand_contact_idxs = generate_matched_random_contacts(
                contacts, rand_contact_pool, n_perm
            )
            # TODO multiprocessing this loop
            for ridx in rand_contact_idxs:
                rand_contacts = rand_contact_pool.iloc[ridx].copy()
                rand_contacts = map_contacts_to_cmap(rand_contacts, res)
                rand_contacts = filter_contacts_by_shape(
                    rand_contacts,
                    cmap_size,
                    n_cntr,
                    n_flnk,
                    fixed_shape=fixed_shape,
                    offdiag=offdiag,
                )
                rand_contacts, rand_submats = aggregate_contacts(
                    cmap,
                    rand_contacts,
                    n_cntr,
                    n_flnk,
                    ddbg=ddbg,
                    min_sparsity=min_sparsity,
                    fixed_shape=fixed_shape,
                    symmetric=symmetric,
                )
                rand_stats = filter_submats_by_stats(
                    calc_submat_stats(rand_submats, n_cntr, n_flnk, save_data=False),
                    fc_type=fc_type,
                    method=filter_opts['method'],
                    n_sigma=filter_opts['n_sigma'],
                )
                permutation_stats[chrom].append(rand_stats)
            del rand_contact_pool
        del cmap

    gw_stats = combine_chrom_stats(obsvd_stats, n_cntr, n_flnk)
    if method == 'perm':
        gw_perm_stats = combine_chrom_stats(
            permutation_stats, n_cntr, n_flnk, n_perm=n_perm
        )
        p_values = calc_permutation_p(gw_stats, gw_perm_stats)
        perm_outputs = prepare_permutation_outputs(out_prfx, n_cntr, n_flnk)
        write_outputs(perm_outputs, gw_perm_stats)
    else:
        p_values = calc_simple_p(gw_stats)

    write_outputs(outputs, gw_stats, p_values=p_values, **plot_opts)


if __name__ == '__main__':
    from docopt import docopt

    args = docopt(__doc__)
    args = {k.lstrip('-<').rstrip('>'): args[k] for k in args}
    try:
        if args.get('debug'):
            logLevel = logging.DEBUG
        elif args.get('quiet'):
            logLevel = logging.WARN
        else:
            logLevel = logging.INFO
        logging.basicConfig(
            level=logLevel,
            format='%(asctime)s; %(levelname)s; %(funcName)s; %(message)s',
            datefmt='%y-%m-%d %H:%M:%S',
        )
        if args.get('prof'):
            import cProfile

            cProfile.run('main(args)')
        else:
            main(args)
    except KeyboardInterrupt:
        logging.warning('Interrupted')
        sys.exit(1)
