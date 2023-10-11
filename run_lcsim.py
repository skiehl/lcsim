#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run blazar light curve simulations.
"""

from math import ceil
import numpy as np
import os

import lcsim
import simdb

#==============================================================================
# CONFIG
#==============================================================================

# initial time sampling parameters:
leakage = 10.
time_sampling = 0.1

# final time sampling parameters:
# TODO

# uncertainty simulation parameters:
# TODO

# PDF scaling:
scaling = 'pdf'

# EMP-style simulation parameters:
n_iter_pdf = 100
keep_non_converged = False
convergence_threshold = 0.01
powerlaw_index_lim = 2.

# check PDF:
check_pdf = True
check_pdf_threshold = 0.05
drop_bad_pdf = True

# number of simulations to store:
n_simulations = {
    'otherwise': 10000}

# number of tries to create the simulations:
n_tries = 2

# number of simulations to create in batch before storing:
n_batch = 200

# number of processes for parallel simulation:
n_processes = 1

# files and directories:
file_psds = 'source_lists/sources_psds.dat'
dir_data = 'data/'
dir_sim = 'sim/'

#==============================================================================
# MAIN
#==============================================================================

# get PSD indices:
cnv = {1: lambda s: float(s.strip() or np.nan)}
sources = np.loadtxt(
        file_psds, dtype=[('name', 'U20'), ('index', float)],
        delimiter=',', usecols=(0, 1), skiprows=1, converters=cnv)

if not sources.shape:
    sources = np.expand_dims(sources, 0)

n_sources = sources.shape[0]

# iterate through sources:
for i, source in enumerate(sources, start=1):
    powerlaw_index = source['index']
    source = source['name'].strip()
    print('\nSource {0:d} of {1:d}: {2:s}'.format(i, n_sources, source))

    # skip if index is not available:
    if np.isnan(powerlaw_index):
        print('No index avaialable. Skip source.')
        continue

    # skip if PSD is too steep for EMP-style simulation:
    if scaling == 'pdf' and powerlaw_index_lim and \
            powerlaw_index > powerlaw_index_lim:
        print(f'Power law index > {powerlaw_index_lim}. Skip source for now.')
        continue

    # load data:
    try:
        dtype = [('mjd', float), ('flux', float), ('flux_err', float)]
        filename = os.path.join(dir_data, f'{source}.csv')
        data = np.loadtxt(filename, delimiter=',', dtype=dtype, skiprows=5)
        time_total = data['mjd'][-1] - data['mjd'][0]
    except OSError:
        print(f'ERROR: {filename} not found.')

    # create directory, if needed:
    if not os.path.isdir(dir_sim):
        os.makedirs(dir_sim)

    # set up data base connection:
    db_file = os.path.join(dir_sim, f'{source}.sqlite3')
    db = simdb.DBConnectorSQLite(db_file)

    # create data base, if needed:
    if not os.path.isfile(db_file):
        db.create_db()

    # get total number of simulations for current source:
    if source in n_simulations.keys():
        n_sim = n_simulations[source]
    else:
        n_sim = n_simulations['otherwise']

    # check how many simulations are still needed:
    n_done = db.number_of_sim()
    n_todo = n_sim - n_done
    print('{0:d} light curves stored.'.format(n_done))

    if n_todo > 0:
        print('{0:d} more light curves will be simulated..\n'.format(n_todo))
    else:
        print('Nothing more to do.\n\n================================')
        continue

    # create simulator instance:
    sim = lcsim.LightCurveSimulator(time_total, time_sampling)
    sim.suggest_time_sampling(data['mjd'], average='median', factor=10.)

    # set spectral shape:
    spec_shape = 'powerlaw'
    # BUGFIX NEEDED: I could also write `sim.powerlaw` instead of `'powerlaw'`, because a callable function works for the simulation code; however this will crash when writing to the database, which expects a string
    spec_args = (powerlaw_index, 10, 10**(-7))
    # note: power of 10. at 10^-7 gives a fairly stable LC amplitude scale,
    # independent of index

    # create counters:
    count_sim = 0
    count_pdf_success = 0
    count_pdf_checks = 0
    count_pdf_accept = 0
    count_pdf_reject = 0
    count_done = 0

    # run the simulations:
    print('\nStarting simulations..')
    adaptive_scaling = 1
    n_iter = ceil(n_todo / n_batch) * n_tries
    i_iter = 0

    while n_todo > 0:
        i_iter += 1

        # abort, if success rate is too low:
        if i_iter > n_iter:
            print('\rProgress: Success rate too low. ABORTED!')
            break

        nlcs = ceil(min(n_batch, n_todo) * adaptive_scaling)
        count_sim += nlcs
        print(f'\rProgress: {nlcs} running, {n_todo} remaining..             ',
              end='')

        pdf = data['flux']

        # Emmanoulopoulos-type simulation:
        if scaling == 'pdf':
            n_created = sim.sim_emp(
                spec_shape, spec_args, pdf, pdf_params=None, pdf_range=None,
                nlcs=nlcs, iterations=n_iter_pdf,
                keep_non_converged=keep_non_converged,
                threshold=convergence_threshold, processes=processes)
            count_pdf_success += n_created

        # Timmer&Koenig-type simulation:
        else:
            n_created = sim.sim_tk(
                spec_shape, spec_args, nlcs=nlcs, processes=processes)
            sim.rescale(pdf.mean(), pdf.std())

        # check PDF:
        if check_pdf:
            n_accept, n_reject = sim.check_pdf(
                    check_pdf_threshold, pdf, drop=drop_bad_pdf)
            count_pdf_checks += n_created
            count_pdf_accept += n_accept
            count_pdf_reject += n_reject

        # iterate though simulations for post-processing and storing:
        for lc in sim.iter_lcs():

            # resample artificial light curves and add observational errors:
            lc.resample(data['mjd'])
            lc.add_errors(data['flux_err'])

            # add simulations to database:
            db.add_sim(lc)
            n_todo -= 1

            # stop if enough light curves have been added to reach goal:
            if n_todo <= 0:
                break

        # update adaptive scaling:
            n_done = sim.number_of_sim()
        count_done += n_done
        success_rate = count_done / count_sim

        if success_rate > 0.5:
            adaptive_scaling = 1 / success_rate

    else:
        print('\rProgress: done.                                             ')

    # print info about success rates:
    print(f'\nSimulations created: {count_sim}')

    if scaling == 'pdf':
        print('PDF adjustment:')
        print('  Success: {0:8d} ({1:.1f}%)'.format(
                count_pdf_success, count_pdf_success/count_sim*100))
        print('  Failure: {0:8d} ({1:.1f}%)'.format(
                count_sim-count_pdf_success,
                (count_sim-count_pdf_success)/count_sim*100))

    if check_pdf:
        print('PDF checks:')
        print('  Accepted: {0:8d} ({1:.1f}%)'.format(
                count_pdf_accept, count_pdf_accept/count_pdf_checks*100))
        print('  Rejected: {0:8d} ({1:.1f}%)'.format(
                count_pdf_reject, count_pdf_reject/count_pdf_checks*100))

    print('\n================================')
