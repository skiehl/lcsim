#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blazar light curve simulation.
"""

from copy import deepcopy
from itertools import repeat
from math import ceil, exp
from multiprocessing import Manager, Pool
import numpy as np
from scipy.stats import kstest
from statsmodels.distributions import ECDF

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD 3"
__version__ = "2.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class ArtificialLightCurve:
    """Process an artificial light curve."""

    #--------------------------------------------------------------------------
    def __init__(self, time, flux, sim_meta={}):
        """Create an instance of ArtificialLightCurve.

        Parameters
        -----
        time : np.ndarray
            Time steps.
        flux : np.ndarray
            Flux values.
        sim_type : str
            Indicating the light curve simulation algorithm.

        TODO
        """

        self.time_orig = time
        self.flux_orig = flux
        self.sim_type = sim_meta['sim_type']

        self.time_orig_total = time[-1] - time[0]
        self.time_orig_sampling = time[1] - time[0]
        self.size = self.time_orig.size

        self.resampled = False
        self.time_res = None
        self.flux_res = None
        self.time_res_total = None
        self.time_res_sampling = None

        self.error_sim = False
        self.flux_err = None
        self.flux_unc = None

        self.sim_meta = sim_meta

    #--------------------------------------------------------------------------
    def __str__(self):
        """Returns gereneral information about the simulated data.
        """

        text = 'Artificial light curve\n'
        text += 'Simulation type:        {0:>10s}\n'.format(self.sim_type)
        text += 'PSD shape:              {0:>10s}\n'.format(
                self.sim_meta['psd_type'])
        text += 'Initial time sampling-------------\n'
        text += 'Time step:              {0:10.3f}\n'.format(
                self.time_orig_sampling)
        text += 'Total time:             {0:10.3f}\n'.format(
                self.time_orig_total)

        if self.resampled == 'const':
            text += 'Resampled-------------------------\n'
            text += 'Time steps:             {0:>10s}\n'.format('even')
            text += 'Time sampling:          {0:10.3f}\n'.format(
                    self.time_res_sampling)
            text += 'Total time:             {0:10.3f}\n'.format(
                    self.time_res_total)
        elif self.resampled in ['powerlaw', 'lognormal', 'ecdf', 'specific']:
            text += 'Resampled-------------------------\n'
            text += 'Time steps:             {0:>10s}\n'.format('uneven')
            text += 'Distribution:           {0:>10s}\n'.format(self.resampled)
            text += 'Median time sampling:   {0:10.3f}\n'.format(
                    self.time_res_sampling)
            text += 'Total time:             {0:10.3f}\n'.format(
                    self.time_res_total)

        if self.error_sim == 'const':
            text += 'Gaussian noise added--------------\n'
            text += 'Uncertainties:  {0:>18s}\n'.format('homoscedastic')
            text += 'Median uncertainty:     {0:10.3f}\n'.format(
                    np.median(self.flux_unc))

        if self.error_sim in ['specific', 'ecdf', 'lognormal']:
            text += 'Gaussian noise added--------------\n'
            text += 'Uncertainties:  {0:>18s}\n'.format('heteroscedastic')
            text += 'Distribution:   {0:>18s}\n'.format(self.error_sim)
            text += 'Median uncertainty:     {0:10.3f}\n'.format(
                    np.median(self.flux_unc))

        if 'pdf_pvalue' in self.sim_meta.keys():
            text += 'PDF checked-----------------------\n'
            text += 'p-value: {0:25.5f}\n'.format(self.sim_meta['pdf_pvalue'])
            text += 'Accepted: {0:>24}\n'.format(
                    str(not self.sim_meta['pdf_reject']))

        return text

    #--------------------------------------------------------------------------
    def _draw_from_powerlaw(self, index, minval, maxval, size=1):
        """Draws random numbers from a truncated power-law distribution.

        Parameters
        -----
        index : float
            Power-law index.
        minval : float
            Lower limit of the distribution.
        maxval : float
            Upper limit of the distribution.
        size : int, default=1
            Number of random data points to return.

        Returns
        -----
        out : np.1darray
            Random numbers.
        """

        index = index + 1.

        return np.power((maxval**index - minval**index) \
                        * np.random.uniform(size=size) + minval**index,
                        1. / index)

    #--------------------------------------------------------------------------
    def _draw_from_ecdf(self, data, size=1):
        """Draws random number from an empirical cumulative distribution
        function (ECDF) defined by given data.

        Parameters
        -----
        data : np.1darray
            The data which defines the ECDF.
        size : int, default=1
            Number of random data points to return.

        Returns
        -----
        out : np.1darray
            Random numbers.
        """

        ecdf = ECDF(data)
        draw = np.random.uniform(low=ecdf.y[1], size=size)

        return np.interp(draw, ecdf.y, ecdf.x)

    #--------------------------------------------------------------------------
    def _random_timesteps(
            self, total_time, dist='powerlaw', params=None, recursion=0):
        """Creates random time data points with time steps following a given
        distribution.

        Parameters
        -----
        total_time : float
            The total time to cover by the time data points
        dist : str, default='powerlaw'
            Defines the distribution the time steps are drawn from. Choose from
            truncated 'powerlaw', 'lognormal' and 'ecdf'. The distribution
            parameters need to be set accordingly in the 'params'.
        params : list
            A list of distribution parameters.
            For 'powerlaw' give (1) the power-law index, (2) the lower, and (3)
            the upper limit of the truncated distribution.
            For 'lognormal' give the distribution (1) mu and (2) sigma.
            For 'ecdf' give an array of time steps (differences between time
            data points not the time data points!).
        recursion : int, default=0
            Do not manually set a value. This parameter is needed internally
            when the drawn time steps are not enough to cover the targeted
            total time and a recursion call of the function is necessery.

        Returns
        -----
        out : np.1darray
            Random time data points.
        """

        # determine number of time steps to create:
        if recursion:
            size = recursion

        elif dist == 'powerlaw' and params[0] < -2.:
            mean_sampling = (params[2]**(params[0] + 2) \
                             - params[1]**(params[0] + 2)) \
                             / (params[2]**(params[0] + 1) \
                                -params[1]**(params[0] + 1)) \
                             * (params[0] + 1) / (params[0] + 2)
            size = int(1.2 * total_time / mean_sampling)
            del mean_sampling

        elif dist=='lognormal':
            mean_sampling = exp((params[0] + params[1]**2) / 2.)
            size = int(1.2 * total_time / mean_sampling)
            del mean_sampling

        elif dist=='ecdf':
            mean_sampling = np.mean(params)
            size = int(1.2 * total_time / mean_sampling)
            del mean_sampling

        else:
            size = 100

        if size < 10:
            size = 10

        # create random time steps:
        if dist=='powerlaw':
            steps = self._draw_from_powerlaw(
                    params[0], params[1], params[2], size=size)

        elif dist=='lognormal':
            steps = np.random.lognormal(
                    mean=params[0], sigma=params[1], size=size)

        elif dist=='ecdf':
            steps = self._draw_from_ecdf(params, size=size)

        else:
            raise ValueError(
                    "Distribution type '{0:s}' is not supported. Either set " \
                    "to 'powerlaw', 'lognormal', or 'ecdf'.".format(dist))

        time = np.cumsum(steps)

        # recursion, if time steps do not cover total time:
        if time[-1] < total_time:
           size = int(ceil(2 * (1 - time[-1] / total_time) * size))
           more = self._random_timesteps(
                   total_time-time[-1], dist=dist, params=params,
                   recursion=size)
           time = np.concatenate((time, more+time[-1]))

        if recursion:
            return time

        time = time[time<=total_time]
        time = np.r_[0, time]

        return time

    #--------------------------------------------------------------------------
    def resample(self, time_steps, params=None):
        """Resample the artificial light curve.

        Parameters
        -----
        time_steps : float, numpy.ndarray, or str
            See notes for details.
        params : list or numpy.ndarray, default=None
            See notes for details.

        Returns
        -----
        None

        Notes
        -----
        This method provides several options to resample the artificial data to
        an even or uneven time grid:
        1. Provide a float for 'time_steps' to resample to an even time grid.
        2. Provide a numpy.ndarray for 'time steps'. The light curve will be
           resampled to these time stamps. The light curve will be cut off,
           when the provided array exceeds the total time of the simulated
           data. Note that the simulated data starts at time 0.
        3. Set 'time_steps' to 'ecdf' and provide a numpy.ndarray of time
           stamps to 'params'. The method will calculate the ECDF of time steps
           (i.e. time differences) between the provided time data and then
           randomly draw time steps from the ECDF to construct a new series of
           time stamps.
        4. Set 'time_steps' to 'powerlaw' and provide a list of three
           parameters to 'param': (1) the power-law index, (2) the lower, and
           (3) the upper limit of the truncated distribution.
           Time steps will be randomly drawn from this distribution to
           construct a new series of time stamps.
        5. Set 'time_steps' to 'lognormal' and provide a list of two parameters
           to 'param': (1) mu and (2) sigma.
           Time steps will be randomly drawn from this distribution to
           construct a new series of time stamps.
        """

        # even time sampling:
        if isinstance(time_steps, float):
            # time step is shorter than original time sampling:
            if time_steps <= self.time_orig_sampling:
                print('Requested time step is shorter or equal to original '
                      'sampling. resample() aborted.')
                return False

            # time step is a multiple of the original sampling:
            ratio = time_steps / self.time_orig_sampling
            if (ratio) % 1. == 0:
                n = int(ratio)
                self.time_res = self.time_orig[::n]
                self.flux_res = self.flux_orig[::n]
                self.resampled = 'const'
                self.time_res_total = self.time_res[-1] - self.time_res[0]
                self.time_res_sampling = time_steps
                self.size = self.time_res.size

                if self.error_sim:
                    self.error_sim = False
                    self.flux_err = None
                    self.flux_unc = None
                    print('Note: resampling removed the error simulation.')

                return True

            # time step requires interpolation:
            self.time_res = np.arange(0, self.time_orig_total, time_steps)
            self.resampled = 'const'

        # resample to specified time steps:
        if isinstance(time_steps, np.ndarray):
            # check that time steps are within total time:
            time_steps -= np.min(time_steps)
            sel = time_steps <= self.time_orig_total
            self.time_res = time_steps[sel]
            self.resampled = 'specific'

        # draw time steps from ECDF of time steps:
        elif time_steps == 'ecdf':
            self.time_res = self._random_timesteps(
                    self.time_orig_total, dist='ecdf', params=params)
            self.resampled = 'ecdf'

        # draw time steps from powerlaw or log-normal distribution:
        elif time_steps in ['powerlaw', 'lognormal']:
            self.time_res = self._random_timesteps(
                    self.time_orig_total, dist=time_steps, params=params)
            self.resampled = time_steps

        else:
            raise ValueError("Unsupported input for 'time_steps'.")

        # interpolate data:
        self.flux_res = np.interp(
                self.time_res, self.time_orig, self.flux_orig)
        self.time_res_total = self.time_res[-1] - self.time_res[0]
        time_diff = np.diff(self.time_res)
        self.time_res_sampling = np.median(time_diff)
        self.size = self.time_res.size

    #--------------------------------------------------------------------------
    def rescale(self, mean, std):
        """Rescale the light curves.

        Parameters
        -----
        mean : float
            The simulated light curves will be shifted such that they have this
            mean value.
        std : float
            The simulated light curves will be scaled such that they have this
            standard deviation.

        Returns
        -----
        None

        Raises
        -----
        Warning
            Raise when the simulated light curves are of Emmanoulopoulos-type.

        Notes
        -----
        The rescaling is applied to the originally sampled and (if applicable)
        to the resampled data. If simulated errors were applied before, those
        do not affect the rescaled data. New errors will be applied after the
        rescaling.
        """

        if self.sim_type == 'EMP':
            raise Warning(
                    "Rescaling Emmanoulopoulos-type light curves is not " \
                    "recommended.")

        # rescale original and resampled data:
        if self.resampled:
            mean_cur = np.mean(self.flux_res)
            std_cur = np.std(self.flux_res)
            scale_factor = std / std_cur
            self.flux_res = (self.flux_res - mean_cur) * scale_factor + mean
            self.flux_orig = (self.flux_orig - mean_cur) * scale_factor + mean

        # rescale original data:
        else:
            mean_cur = np.mean(self.flux_orig)
            std_cur = np.std(self.flux_orig)
            scale_factor = std / std_cur
            self.flux_orig = (self.flux_orig - mean_cur) * scale_factor + mean

        # apply new error simulation:
        if self.error_sim:
            self._add_errors()

    #--------------------------------------------------------------------------
    def _add_errors(self):
        """Draws random Gaussian errors and adds them to simulated data.

        Returns
        -----
        None
        """

        err = np.random.normal(loc=0, scale=self.flux_unc, size=self.size)

        if self.resampled:
            self.flux_err = self.flux_res + err
        else:
            self.flux_err = self.flux_orig + err

    #--------------------------------------------------------------------------
    def add_errors(self, uncertainties, params=None):
        """Add Gaussian noise to the artificial light curve.

        An error term is drawn randomly for each data point from a Gaussian
        distribution and added to the simulated data:

        .. math::
            f_{\mathrm{sim},i} \\rightarrow f_{\mathrm{sim},i} + f_{\mathrm{err},i}

        with


        .. math::
            f_{\mathrm{err},i} \sim \\mathcal{N}(0, \sigma_i)

        The noise scale sigma_i can be be the same for all data points or vary.

        Parameters
        -----
        uncertainties : float, numpy.ndarray, or str
            See notes for details.
        params : list or numpy.ndarray, default=None
            See notes for details.

        Returns
        -----
        None

        Notes
        -----
        This method provides several options to add homoscedastic and
        heteroscedastic error to the artificial data:
        1. Provide a float for 'uncertainties' to use the same uncertainty
           scale for each data point (homoscedasticity).
        2. Provide a numpy.ndarray for 'uncertainties'. The values in the array
           are used as the uncertainty scales for each corresponding data
           point. Note that the length of 'uncertainties' needs to match the
           length of the simulated light curve.
        3. Set 'uncertainties' to 'ecdf' and provide a numpy.ndarray of
           uncertainties to 'params'. The method will calculate the ECDF of the
           uncertainties and then randomly draw uncertainties from the ECDF.
        4. Set 'uncertainties' to 'lognormal' and provide a list of two
           parameters to 'param': (1) mu and (2) sigma.
           Uncertainties are randomly drawn from this distribution.
        """

        # homoscedastic uncertainties:
        if isinstance(uncertainties, float):
            self.flux_unc = np.ones(self.size) * uncertainties
            self.error_sim = 'const'

        # specific uncertainties:
        elif isinstance(uncertainties, np.ndarray) \
                and uncertainties.size == self.size:
            self.flux_unc = uncertainties
            self.error_sim = 'specific'

        # draw uncertainties from ECDF of uncertainties:
        elif uncertainties == 'ecdf':
            self.flux_unc = self._draw_from_ecdf(params, size=self.size)
            self.error_sim = 'ecdf'

        # draw uncertainties from log-normal distribution:
        elif uncertainties == 'lognormal':
            self.flux_unc = np.random.lognormal(
                    params[0], params[1], size=self.size)
            self.error_sim = 'lognormal'

        else:
            raise ValueError("Unsupported input for 'uncertainties'.")

        self._add_errors()

    #--------------------------------------------------------------------------
    def data(self, get_all=False):
        """Get the simulated light curve data.

        Parameters
        -----
        get_all : bool, default=False
            As default returns only the final light curve data, i.e. the
            resampled and/or noise-added light curve.
            If True, returns all data including the original sampling.

        Returns
        -----
        out : dict
            Simulated light curve data.
        """

        # return all light curves (original, resampled, errors added):
        if get_all:
            # add original light curve:
            results = {
                    'time_orig': self.time_orig,
                    'flux_orig': self.flux_orig}

            # add resampled light curve (if available):
            if self.resampled:
                results['time_res'] = self.time_res
                results['flux_res'] = self.flux_res

            # add error-added light curve (if available):
            if self.error_sim:
                results['flux_err'] = self.flux_err
                results['flux_unc'] = self.flux_unc

        # return final light curve (resampled and error-added):
        elif self.error_sim and self.resampled:
            results = {
                    'time': self.time_res,
                    'flux': self.flux_err,
                    'flux_unc': self.flux_unc}

        # return final light curve (error-added):
        elif self.error_sim:
            results = {
                    'time': self.time_orig,
                    'flux': self.flux_err,
                    'flux_unc': self.flux_unc}

        # return final light curve (resampled):
        elif self.resampled:
            results = {
                    'time': self.time_res,
                    'flux': self.flux_res}

        # return final light curve (original):
        else:
            results = {
                    'time': self.time_orig,
                    'flux': self.flux_orig}

        return results

#==============================================================================

class LightCurveSimulator:
    """Simulate blazar light curves as a random noise process."""

    #--------------------------------------------------------------------------
    def __init__(self, time_total, time_sampling, leakage=10):
        """Create an instance of LightCurveSimulator.

        Parameters
        -----
        time_total : float
            Set the total time for the simulated data.
        time_sampling : float
            Set the time sampling for the simulated data.
        leakage : float, default=10
            The total time of a simulation will be increased by this factor,
            which must be larger than 1. This will include power at
            frequencies lower than that corresponding to the total time in the
            simulated data.
        """

        self.set_time_sampling(time_total, time_sampling, leakage=leakage)
        self.lightcurves = []

    #--------------------------------------------------------------------------
    def number_of_sim(self):
        """Return the number of currently stored simulations.

        Returns
        -------
        int
            Number of currently stored simulations.
        """

        return len(self.lightcurves)

    #--------------------------------------------------------------------------
    def set_time_sampling(self, time_total, time_sampling, leakage=10):
        """Create an instance of LightCurveSimulator.

        Parameters
        -----
        time_total : float
            Set the total time for the simulated data.
        time_sampling : float
            Set the time sampling for the simulated data.
        leakage : float, default=10
            The total time of a simulation will be increased by this factor,
            which must be larger than 1. This will include power at
            frequencies lower than that corresponding to the total time in the
            simulated data.

        Raises
        -----
        ValueError
            Raised, if time_total, time_sampling, or leakage is no float-like.
            Raised, if leakage is smaller than 1.

        Returns
        -----
        None
        """

        # check inputs:
        try:
            time_total = float(time_total)
        except:
            raise ValueError("'time_total' must be float-like.")

        try:
            time_sampling = float(time_sampling)
        except:
            raise ValueError("'time_sampling' must be float-like.")

        try:
            leakage = float(leakage)
        except:
            raise ValueError("'leakage' must be float-like.")

        if leakage < 1:
            raise ValueError("'leakage' must be equal to or larger than 1.")

        # save parameters:
        self.time_total = time_total
        self.time_sampling = time_sampling
        self.leakage = leakage

        # number of data points for initial simulation:
        ndp_init = int(ceil(time_total * leakage / time_sampling)) + 1
        self.ndp_init = ndp_init

        # number of data points and time steps for final simulation(s):
        ndp = int(ceil(time_total / time_sampling)) + 1
        time_total = time_sampling * (ndp - 1)
        time = np.linspace(0, time_total, ndp)
        self.ndp = ndp
        self.time = time

    #--------------------------------------------------------------------------
    @staticmethod
    def suggest_time_sampling(time, average='median', factor=10.):
        """Suggests a total time and time sampling for simulated data, based on
        input time data. The total time is increased by a given factor, the
        time sampling reduced by the same factor, to include low and high
        frequency power in the simulated noise process.

        Parameters
        -----
        time : 1darray
            Time series.
        average : string, default='median'
            Choose average type ('median' or 'mean') for suggested sampling.
        factor : float, default=10.
            The suggested total time and sampling are modified by this factor.

        Returns
        -----
        out, out : float, float
            Suggested (minimum) total time and (maximum) sampling rate.
        """

        total_time = time[-1] - time[0]
        deltat = np.diff(time)
        sampling_median = np.median(deltat)
        sampling_mean = np.mean(deltat)
        sampling_min = np.min(deltat)
        sampling_max = np.max(deltat)

        if average=='median':
            sim_sampling = sampling_median / factor
        elif average=='mean':
            sim_sampling = sampling_mean / factor
        sim_total = total_time * factor

        print('Total time:         {0:8.3f}'.format(total_time))
        print('Min. sampling:      {0:8.3f}'.format(sampling_min))
        print('Max. sampling:      {0:8.3f}'.format(sampling_max))
        print('Mean sampling:      {0:8.3f}'.format(sampling_mean))
        print('Median sampling:    {0:8.3f}'.format(sampling_median))
        print('Suggested')
        print('Maximum sampling:   {0:8.3f}'.format(sim_sampling))
        print('Minimum total time: {0:8.3f}'.format(sim_total))

        return sim_total, sim_sampling

    #--------------------------------------------------------------------------
    def powerlaw(self, frequencies, index=1., amplitude=10., frequency=0.1):
        """Returns an array of amplitudes following a power-law over the input
        frequencies.

        Parameters
        -----
        frequencies : 1darray
            Frequencies for which to calculate the power-law in arbitrary
            units.
        index : float, default=1.
            Power-law index.
        amplitude : float, default=10.
            Power-law amplitude at 'frequency' in arbitrary unit.
        frequency : float, default=0.1
            Frequency for the given 'amplitude' in same unit as 'frequencies'.

        Returns
        -----
        out : 1darray
            Array of same length as input 'frequencies'.

        Notes
        -----
        Can be used as a generic shape for the power spectrum of a simulated
        light curve.
        """

        return amplitude * np.power(frequencies / frequency, -index)

    #--------------------------------------------------------------------------
    def kneemodel(self, frequencies, index=1., amplitude=10., frequency=0.1):
        """Returns an array of amplitudes following a constant profile that
        changes into a power-law around a given frequency.

        Parameters
        -----
        frequencies : 1darray
            Frequencies for which to calculate the power-law in arbitrary
            units.
        index : float, default=1.
            Power-law index.
        amplitude : float, default=10.
            Constant amplitude at frequencies below 'frequency' in arbitrary
            unit.
        frequency : float, default=0.1
            Frequency  in same unit as 'frequencies' at which profile changes
            into a power-law.

        Returns
        -----
        out : 1darray
            Array of same length as input 'frequencies'.

        Notes
        -----
        Can be used as a generic shape for the power spectrum of a simulated
        light curve.
        """

        return amplitude * np.power(1 + np.power(frequencies / frequency, 2),
                                    -index / 2.)

    #--------------------------------------------------------------------------
    def brokenpowerlaw(
            self, frequencies, index_lo=1., index_hi=2., amplitude=10.,
            frequency=0.1):
        """Returns an array of amplitudes following a broken power-law.

        Parameters
        -----
        frequencies : array
            Frequencies for which to calculate the power-law in arbitrary units.
        index_hi : float, default=2.
            Power-law index at frequencies lower than 'frequency'.
        index_lo : float, default=1.
            Power-law index at frequencies higher than 'frequency'.
        frequency : float, default=0.1
            Frequency of the power-law break in same unit as 'frequencies'.
        amplitude : float, default=10.
            Amplitude at 'frequency' in arbitrary unit.

        Returns
        -----
            Array of same length as input 'frequencies'.

        Notes
        -----
        Can be used as a generic shape for the power spectrum of a simulated
        light curve.
        """

        return np.where(
                frequencies > frequency,
                amplitude * np.power(frequencies / frequency, -index_hi),
                amplitude * np.power(frequencies / frequency, -index_lo))

    #--------------------------------------------------------------------------
    def multi_logn(self, x, *params):
        """Multi-component log-normal function.

        Returns the function values at position x.

        Parameters
        ------
        x : float or np.ndarray
            Function is evaluated at these positions.
        params : list
            Function parameters. For each log-normal component three
            parameters: peak position, peak amplitude, peak width. For multiple
            components append tripplets of these parameters (e.g.
            [pos1, amp1, wid1, pos2, amp2, wid2]).

        Returns
        -----
        out : float or np.ndarray
            Function evaluated at x.

        Notes
        -----
        Can be used as a generic shape for the probability density of a
        simulated light curve.
        """

        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            pos = params[i]
            amp = params[i+1]
            wid = params[i+2]
            y = y + amp * np.exp( -((np.log(x) - np.log(pos)) / wid)**2)

        return y

    #--------------------------------------------------------------------------
    def _draw_from_func(self, func, params, x_min, x_max, size=1, seed=False):
        """Draws random number from a function.

        Parameters
        -----
        func : function
            A function to draw random mumbers from.
        params : list
            The parameters of the PDF.
        x_min : float
            The minimum value to draw.
        x_min : float
            The maximum value to draw.
        size : int, default=1
            Number of random data points to return.
        seed : int, default=False
            Sets a seed for the random generator to get a reproducable result.
            For testing only.

        Returns
        -----
        out : np.1darray
            Random numbers.

        Notes
        -----
        Can be used to draw random numbers from a probability density function.
        """

        x = np.linspace(x_min, x_max, 1000)
        pdf = func(x, *params)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        if seed:
            np.random.seed(seed)
        rand = np.random.uniform(0, 1, size)
        rand = np.interp(rand, cdf, x)

        return rand

    #--------------------------------------------------------------------------
    def _iterations(self, nlcs):
        """Calculate how many simulations are needed to create a given number
        of light curves.

        nlcs : int
            Number of light curves

        Parameters
        -----
        nlcs : int
            Number of light curves.

        Returns
        -----
        out, out : int, int
            Number of simulations needed to create the intended number of
            light curves.
            Number of light curves that can be extracted from one iteration.
        """

        nlcs_per_iter = (self.ndp_init - 1) // (self.ndp - 1)
        n_iter = ceil(nlcs / nlcs_per_iter)

        return n_iter, nlcs_per_iter

    #--------------------------------------------------------------------------
    def _split_lc(self, lightcurve, nlcs_per_iter):
        """Split long light curve into segments.

        Parameters
        ----------
        lightcurve : numpy.ndarray
            Simulated light curve.
        nlcs_per_iter : int
            Number of equally sized segments that the input light curve should
            be split into.

        Returns
        -------
        lightcurves : list of numpy.ndarray
            The light curve segments.
        """

        lightcurves = []

        for i in range(nlcs_per_iter):
            m = i * (self.ndp - 1)
            n = (i + 1) * (self.ndp - 1) + 1
            lightcurves.append(lightcurve[m:n])

        return lightcurves

    #--------------------------------------------------------------------------
    def _sim_tk(self, spec_shape, spec_args, n_splits=1, seed=False):
        """Create Gaussian noise with a given spectrum.

        This function implements the Timmer & Koenig, 1995 [1] algorithm for
        producing an artificial light curve. See docstring of
        self.sim_tk() for details. This is a helper function that is called by
        self._wrapper_tk().

        Parameters
        -----
        spec_shape : func
            Function that takes an array of frequencies and 'spec_args' as
            input and calculates a spectrum for those frequencies.
        spec_args : list
            Function arguments to 'spec_shape'.
        seed : int, default=False
            Sets a seed for the random generator to get a reproducable result.
            For testing only.

        Returns
        -----
        out : numpy.ndarray or list
            Simulated light curve following a random noise process with the
            input power spectrum.
            If n_splits>1, a list of light curve segments is returned.

        References
        -----
        [1] Timmer and Koenig, 1995, 'On generating power law noise', A&A, 300,
            707
        """

        # set spectrum:
        freq = np.fft.rfftfreq(self.ndp_init, self.time_sampling)
        freq[0] = 1
        if not callable(spec_shape):
            spec_shape = eval('self.{0:s}'.format(spec_shape))
        spectrum = spec_shape(freq[1:], *spec_args)
        spectrum[0] = 0
        del freq

        # random (complex) Fourier coefficients for inverse Fourier transform:
        if seed:
            np.random.seed(seed)
        coef = np.random.normal(size=(2, spectrum.shape[0]))

        # if N is even the Nyquist frequency is real:
        if self.ndp_init % 2 == 0:
            coef[-1,1] = 0.

        # complex coefficients:
        coef = coef[0] + 1j * coef[1]

        # scale coefficients with spectrum:
        coef *= np.sqrt(0.5 * spectrum * self.ndp_init)
        # above line is what works to get correct PSD slope, differs from
        # definition in T&K95, see argument in PhD notebook
        # 1.2_AmplitudeProblem.ipynb
        coef *= 10**(spec_args[0] * 2.5)
        # 2.5 is an empirical scaling factor to get the correct amplitude
        # (approximately; would be better to understand where this is coming
        # from)

        # inverse Fourier transform:
        lightcurve = np.fft.irfft(coef, self.ndp_init)

        # normalize to zero mean:
        lightcurve -= np.mean(lightcurve)

        return lightcurve

    #--------------------------------------------------------------------------
    def _wrapper_tk(
            self, __, queue, spec_shape, spec_args, n_splits=1, seed=False):
        """Helper function that is called by sim_tk() process pool to simulate
        light curves and store them in a queue.

        Parameters
        -----
        __ : range
            Iterator for the repetition of the lightcurve simulation. Only
            required for the parallel processing. Not required for this
            function.
        queue : multiprocessing.managers.AutoProxy[Queue]
            Queue shared among processes for saving lightcurves.
        spec_shape : func
            Function that takes an array of frequencies and 'spec_args' as
            input and calculates a spectrum for those frequencies.
        spec_args : list
            Function arguments to 'spec_shape'.
        n_splits : int, default=1
            Number of equally sized segments that the light curve will be
            split into.
        seed : int, default=False
            Sets a seed for the random generator to get a reproducable result.
            For testing only.

        Returns
        -----
        None
        """

        # create light curve:
        lightcurve = self._sim_tk(spec_shape, spec_args, seed=seed)

        # split into segements:
        if n_splits > 1:
            lightcurve = self._split_lc(lightcurve, n_splits)
        else:
            lightcurve = [lightcurve]

        # store in queue:
        for lc in lightcurve:
            queue.put(lc)

    #--------------------------------------------------------------------------
    def sim_tk(self, spec_shape, spec_args, nlcs=1, processes=1, seed=False):
        """Simulates one/multiple equally sampled light curve(s) following a
        random noise process with given a spectral shape of the power spectral
        density [1].

        Parameters
        -----
        spec_shape : func
            Function that takes an array of frequencies and 'spec_args' as
            input and calculates a spectrum for those frequencies.
        spec_args : list
            Function arguments to 'spec_shape'.
        nlcs : int, default=1
            Set the number of light curves that are produced. See notes below.
        processes : int, default=1
            Number of processes used to simulate light curves.
        seed : int, default=False
            Sets a seed for the random generator to get a reproducable result.
            For testing only.

        Returns
        -----
        int
            Number of simulated light curves.

        Notes
        -----
        The method will initially produce one long light curve with a total
        time N times longer than the final total time, where N is set by
        'nlcs'. This long light curve is then split into N individual pieces.
        Producing one initial long light curves means that power at frequencies
        N times lower than the final total time is included in the noise
        process. Setting ncls>1 allows to include low-frequency power.

        References
        -----
        [1] Timmer and Koenig, 1995, 'On generating power law noise', A&A, 300,
            707
        """

        # parallel simulation of light curves:
        n_iter, nlcs_per_iter = self._iterations(nlcs)
        manager = Manager()
        queue = manager.Queue()

        if n_iter < processes:
            processes = n_iter

        with Pool(processes=processes) as pool:
            pool.starmap(
                    self._wrapper_tk,
                    zip(range(n_iter), repeat(queue), repeat(spec_shape),
                        repeat(spec_args), repeat(nlcs_per_iter),
                        repeat(seed)))

        # extract light curves from queue:
        self.lightcurves = []

        while len(self.lightcurves) < nlcs:
            self.lightcurves.append(queue.get())

        self.sim_type = 'TK'
        self.lc_scale = 'None'
        self.psd_type = spec_shape
        self.pdf_check = False

        return len(self.lightcurves)

    #--------------------------------------------------------------------------
    def rescale(self, mean, std):
        """Rescale the light curves.

        Parameters
        -----
        mean : float
            The simulated light curves will be shifted such that they have this
            mean value.
        std : float
            The simulated light curves will be scaled such that they have this
            standard deviation.

        Returns
        -----
        None

        Raises
        -----
        Warning
            Raise when the simulated light curves are of Emmanoulopoulos-type.
        """

        if self.sim_type == 'EMP':
            raise Warning(
                    "Rescaling Emmanoulopoulos-type light curves is not " \
                    "recommended.")

        for i, lightcurve in enumerate(self.lightcurves):
            mean_cur = lightcurve.mean()
            std_cur = lightcurve.std()
            scale_factor = std / std_cur
            self.lightcurves[i] = (lightcurve - mean_cur) * scale_factor + mean

    #--------------------------------------------------------------------------
    def _adjust_pdf(
            self, lightcurve, pdf, pdf_params=None, pdf_range=None,
            iterations=100, keep_non_converged=False, threshold=0.01):
        """Change the PDF of a light curve while maintaining its PSD.
        This function implements the Emmanoulopoulos et al, 2013 [1] algorithm.
        See docstring of self.adjust_pdf() for details.
        This is a helper function that is called by self._wrapper_pdf().

        Parameters
        -----
        lightcurve : numpy.ndarray
            Flux density data of an evenly sampled light curve.
        pdf : numpy.ndarray or callable
            When providing flux data in a numpy.array the method will calculate
            an ECDF of the input data and draw random flux values from that
            ECDF.
            When providing a callable function the method will use that
            function to draw random flux values. Parameters for the function
            need to be provided in 'pdf_params'
        pdf_params : list, default=None
            When a callable function is given to 'pdf', corresponding
            parameters need to be given to 'pdf_params'.
        pdf_range : list
            List of two elements. Random flux values will be drawn from the
            given PDF between the two limits provided by 'pdf_range'. Required
            only when a callable function is given to 'pdf'. Does not have an
            effect, when a numpy.ndarray is given to 'pdf'.
        iterations : int, default=100
            The algorithm [1] is iterative. This value sets a maximum number of
            iterations to avoid infinite loops.
        keep_non_converged : bool or str, default=False
            If True the result is returned even if the algorithm [1] did not
            converge.
            If False and the algorithm [1] did not converge, False is returned.
            If set to 'ask', a notification is written asking how to proceed.
        threshold : float, default=0.01
            Defines when the algorithm [1] is considered as converged.
            Iterations are stopped when maximum difference between the current
            and the previous light curve does not exceed this 'threshold'.

        Returns
        -----
        out : numpy.ndarray
            Simulated light curve with the target PDF.
        out : int
            Number of iterations reached until convergence.
        out : bool
            True, if converged. False, otherwise.

        References
        -----
        [1] Emmanoulopoulos et al, 2013, MNRAS, 433, 907
        """

        # discrete Fourier transform:
        dft_norm = np.fft.rfft(lightcurve)
        ampl_adj = np.absolute(dft_norm)

        # create artificial light curve based on ECDF of input data:
        if isinstance(pdf, np.ndarray):
            ecdf = ECDF(pdf)
            lc_sim = np.interp(np.random.uniform(ecdf.y[1], 1., size=self.ndp),
                               ecdf.y, ecdf.x)

        # or create artificial light curve based on model PDF:
        elif callable(pdf):
            lc_sim = self._draw_from_func(
                    pdf, pdf_params, pdf_range[0], pdf_range[1], size=self.ndp)

        else:
            raise ValueError(
                    "'pdf' needs to be a np.ndarray or a function.")

        converged = False

        # iteration:
        for i in range(iterations):
            # calculate DFT, amplitudes:
            dft_sim = np.fft.rfft(lc_sim)
            ampl_sim = np.absolute(dft_sim)

            # spectral adjustment:
            dft_adj = dft_sim / ampl_sim * ampl_adj
            lc_adj = np.fft.irfft(dft_adj, n=self.ndp)

            # amplitude adjustment:
            a = np.argsort(lc_adj)
            s = np.argsort(lc_sim)
            lc_adj[a] = lc_sim[s]

            # check if process converged:
            if np.max(np.absolute(lc_adj -lc_sim) / lc_sim) < threshold:
                converged = True
                break
            else:
                lc_sim = deepcopy(lc_adj)

        # no convergence reached:
        else:
            # ask what to do:
            if keep_non_converged == 'ask':
                inp = input(
                        'No convergence reached within {0:d} iterations. ' \
                        'Keep (y), throw away (n), or try again (r)?'.format(
                                iterations))
                if inp == 'y':
                    pass
                elif inp == 'r':
                    lc_sim, i, converged = self._adjust_pdf(
                            lightcurve, pdf, pdf_params=pdf_params,
                            pdf_range=pdf_range, iterations=iterations,
                            keep_non_converged=keep_non_converged,
                            threshold=threshold)
                else:
                    lc_sim = False

            # keep result anyway:
            elif keep_non_converged:
                pass

            # return False:
            else:
                lc_sim = False

        return lc_sim, i, converged

    #--------------------------------------------------------------------------
    def _wrapper_pdf(
            self, queue, lightcurve, pdf, pdf_params=None, pdf_range=None,
            iterations=100, keep_non_converged=False, threshold=0.01):
        """Helper function that is called by adjust_pdf() process pool to
        change the PDF of light curves and store the results in a queue.

        Parameters
        -----
        queue : multiprocessing.managers.AutoProxy[Queue]
            Queue shared among processes for saving lightcurves.
        lightcurve : numpy.ndarray
            Flux density data of an evenly sampled light curve.
        pdf : numpy.ndarray or callable
            When providing flux data in a numpy.array the method will calculate
            an ECDF of the input data and draw random flux values from that
            ECDF.
            When providing a callable function the method will use that
            function to draw random flux values. Parameters for the function
            need to be provided in 'pdf_params'
        pdf_params : list, default=None
            When a callable function is given to 'pdf', corresponding
            parameters need to be given to 'pdf_params'.
        pdf_range : list
            List of two elements. Random flux values will be drawn from the
            given PDF between the two limits provided by 'pdf_range'. Required
            only when a callable function is given to 'pdf'. Does not have an
            effect, when a numpy.ndarray is given to 'pdf'.
        iterations : int, default=100
            The algorithm [1] is iterative. This value sets a maximum number of
            iterations to avoid infinite loops.
        keep_non_converged : bool or str, default=False
            If True the result is returned even if the algorithm [1] did not
            converge.
            If False and the algorithm [1] did not converge, False is returned.
            If set to 'ask', a notification is written asking how to proceed.
        threshold : float, default=0.01
            Defines when the algorithm [1] is considered as converged.
            Iterations are stopped when maximum difference between the current
            and the previous light curve does not exceed this 'threshold'.

        Returns
        -----
        None
        """

        lc_sim, i, converged = self._adjust_pdf(
                lightcurve, pdf, pdf_params=pdf_params, pdf_range=pdf_range,
                iterations=iterations, keep_non_converged=keep_non_converged,
                threshold=threshold)
        queue.put((lc_sim, i, converged))

    #--------------------------------------------------------------------------
    def adjust_pdf(
            self, pdf, pdf_params=None, pdf_range=None, iterations=100,
            keep_non_converged=False, threshold=0.01, processes=1):
        """Change the PDF of all simulated light curves to a target PDF.

        This method implements the Emmanoulopoulos et al, 2013 [1] algorithm
        for changing the PDF of a light curve while maintaining its PSD. This
        routine is automatically applied to all simulated light curves stored
        in the an LightCurveSimulator-instance.

        Parameters
        -----
        pdf : numpy.ndarray or callable
            When providing flux data in a numpy.array the method will calculate
            an ECDF of the input data and draw random flux values from that
            ECDF.
            When providing a callable function the method will use that
            function to draw random flux values. Parameters for the function
            need to be provided in 'pdf_params'
        pdf_params : list, default=None
            When a callable function is given to 'pdf', corresponding
            parameters need to be given to 'pdf_params'.
        pdf_range : list
            List of two elements. Random flux values will be drawn from the
            given PDF between the two limits provided by 'pdf_range'. Required
            only when a callable function is given to 'pdf'. Does not have an
            effect, when a numpy.ndarray is given to 'pdf'.
        iterations : int, default=100
            The algorithm [1] is iterative. This value sets a maximum number of
            iterations to avoid infinite loops.
        keep_non_converged : bool or str, default=False
            If True the result is returned even if the algorithm [1] did not
            converge.
            If False and the algorithm [1] did not converge, False is returned.
            If set to 'ask', a notification is written asking how to proceed.
        threshold : float, default=0.01
            Defines when the algorithm [1] is considered as converged.
            Iterations are stopped when maximum difference between the current
            and the previous light curve does not exceed this 'threshold'.
        processes : int, default=1
            Number of processes used to simulate light curves.

        Returns
        -----
        None

        References
        -----
        [1] Emmanoulopoulos et al, 2013, MNRAS, 433, 907
        """

        # check that the PDFs have not been adjusted yet:
        if self.sim_type == 'EMP':
            print('Light curve PDFs have already been adjusted. ' \
                  'adjust_pdf() aborted!')
            return None

        # parallel processing of light curves:
        manager = Manager()
        queue = manager.Queue()

        if len(self.lightcurves) < processes:
            processes = len(self.lightcurves)

        with Pool(processes=processes) as pool:
            pool.starmap(
                    self._wrapper_pdf,
                    zip(repeat(queue), self.lightcurves, repeat(pdf),
                        repeat(pdf_params), repeat(pdf_range),
                        repeat(iterations), repeat(keep_non_converged),
                        repeat(threshold)))

        # extract results from queue:
        nlcs = len(self.lightcurves)
        self.lightcurves = []
        self.pdf_n_iter = []
        self.pdf_converged = []

        for i in range(nlcs):
            lc_adj, n_iter, converged = queue.get()

            if lc_adj is not False:
                self.lightcurves.append(lc_adj)
                self.pdf_n_iter.append(n_iter)
                self.pdf_converged.append(converged)

        # remove non-converged light curves:
        if not keep_non_converged:
            self.lightcurves = [
                    lc for lc in self.lightcurves if lc is not False]
            self.pdf_n_iter = [
                    n_iter for n_iter, converged \
                    in zip(self.pdf_n_iter, self.pdf_converged) if converged]
            self.pdf_converged = [True] * len(self.pdf_n_iter)

        self.sim_type = 'EMP'
        if isinstance(pdf, ECDF):
            self.lc_scale = 'ECDF'
        elif callable(pdf):
            self.lc_scale = 'PDF'
        else:
            self.lc_scale = False

    #--------------------------------------------------------------------------
    def sim_emp(
            self, spec_shape, spec_args, pdf, pdf_params=None, pdf_range=None,
            nlcs=1,  iterations=100, keep_non_converged=False, threshold=0.01,
            processes=1):
        """Create a simulated light curve with a specified PSD and PDF.

        This method combines the two steps of producting a Gaussian light curve
        with a given PSD [1] and adjusting its PDF to a specified PDF [2].

        Parameters
        -----
        spec_shape : func
            Function that takes an array of frequencies and 'spec_args' as
            input and calculates a spectrum for those frequencies.
        spec_args : list
            Function arguments to 'spec_shape'.
        pdf : numpy.ndarray or callable
            When providing flux data in a numpy.array the method will calculate
            an ECDF of the input data and draw random flux values from that
            ECDF.
            When providing a callable function the method will use that
            function to draw random flux values. Parameters for the function
            need to be provided in 'pdf_params'
        pdf_params : list, default=None
            When a callable function is given to 'pdf', corresponding
            parameters need to be given to 'pdf_params'.
        pdf_range : list
            List of two elements. Random flux values will be drawn from the
            given PDF between the two limits provided by 'pdf_range'. Required
            only when a callable function is given to 'pdf'. Does not have an
            effect, when a numpy.ndarray is given to 'pdf'.
        nlcs : int, default=1
            Set the number of light curves that are produced. See notes below.
        iterations : int, default=100
            The algorithm [1] is iterative. This value sets a maximum number of
            iterations to avoid infinite loops.
        keep_non_converged : bool or str, default=False
            If True the result is returned even if the algorithm [1] did not
            converge.
            If False and the algorithm [1] did not converge, False is returned.
            If set to 'ask', a notification is written asking how to proceed.
        threshold : float, default=0.01
            Defines when the algorithm [1] is considered as converged.
            Iterations are stopped when maximum difference between the current
            and the previous light curve does not exceed this 'threshold'.
        processes : int, default=1
            Number of processes used to simulate light curves.

        Returns
        -----
        int
            Number of simulated light curves.

        Notes
        -----
        The method will initially produce one long Gaussian light curve with a
        total time N times longer than the final total time, where N is set by
        'nlcs'. This long light curve is then split into N individual pieces.
        Producing one initial long light curves means that power at frequencies
        N times lower than the final total time is included in the noise
        process. Setting ncls>1 allows to include low-frequency power.
        The adjustment of the PDF is then applied to every split light curve
        individually.

        References
        -----
        [1] Timmer and Koenig, 1995, 'On generating power law noise', A&A, 300,
            707
        [2] Emmanoulopoulos et al, 2013, MNRAS, 433, 907
        """

        self.sim_tk(spec_shape, spec_args, nlcs=nlcs, processes=processes)
        self.adjust_pdf(
                pdf, pdf_params=pdf_params, pdf_range=pdf_range,
                iterations=iterations, keep_non_converged=keep_non_converged,
                threshold=threshold, processes=processes)
        self.pdf_check = False

        return len(self.lightcurves)

    #--------------------------------------------------------------------------
    def check_pdf(self, threshold, cdf, args=(), drop=False, verbose=0):
        """Check the CDF of the simulated light curves against a reference CDF.

        Parameters
        ----------
        threshold : float
            The p-value threshold for rejecting a simulation.
        cdf : str, array_like, or callable
            If array_like, it should be a 1-D array of observations of random
            variables. If a callable, that callable is used to calculate the
            cdf. If a string, it should be the name of a distribution in
            scipy.stats, which will be used as the cdf function.
        args : tuple, sequence, default=()
            Distribution parameters, used if 'cdf' is string or callable.
        drop : bool, default=False
            If True, simulations that are rejected are deleted. Otherwise,
            simulations are just flagged.

        Returns
        -------
        n_accept : int
            Number of accepted light curves.
        n_reject : int
            Number of rejected light curves.
        """

        n_lcs = len(self.lightcurves)
        self.pdf_check = True
        self.pdf_threshold = threshold
        self.pdf_pvalues = np.zeros(n_lcs)
        self.pdf_reject = np.zeros(n_lcs, dtype=bool)

        for i, lightcurve in enumerate(self.lightcurves):
            pvalue = kstest(lightcurve, cdf, args=args).pvalue
            reject = pvalue < threshold
            self.pdf_pvalues[i] = pvalue
            self.pdf_reject[i] = reject


        n_reject = np.sum(self.pdf_reject)
        n_accept = n_lcs - n_reject
        self.pdf_accept_rate = n_accept / n_lcs

        if drop:
            for i in np.nonzero(self.pdf_reject)[0][::-1]:
                del self.lightcurves[i]

            sel = ~self.pdf_reject
            self.pdf_pvalues = self.pdf_pvalues[sel]
            self.pdf_reject = self.pdf_reject[sel]

        if verbose:
            info = f'{n_reject} LCs rejected out of {n_lcs}. Acceptance ' \
                   f'rate: {self.pdf_accept_rate*100:.1f} %.'
            print(info)

        return n_accept, n_reject

    #--------------------------------------------------------------------------
    def iter_lcs(self):
        """Iterate through simulated light curves.

        Parameters
        -----
        None

        Yields
        -----
        out : ArtificialLightCurve
            Simulated light curve.
        """

        for i, flux in enumerate(self.lightcurves):

            # prepare simulation metadata:
            sim_meta = {'sim_type': self.sim_type, 'psd_type': self.psd_type}

            if self.sim_type == 'EMP':
                sim_meta['pdf_n_iter'] = self.pdf_n_iter[i]
                sim_meta['pdf_converged'] = self.pdf_converged[i]

            if self.pdf_check:
                sim_meta['pdf_pvalue'] = self.pdf_pvalues[i]
                sim_meta['pdf_reject'] = self.pdf_reject[i]


            lightcurve = ArtificialLightCurve(
                    self.time, flux, sim_meta=sim_meta)

            yield lightcurve

    #--------------------------------------------------------------------------
    def get_lcs(self):
        """Returns the simulated light curves.

        Parameters
        -----
        None

        Returns
        -----
        out : list
            List of simulated light curves, each an ArtificialLightCurve
            instance.
        """

        return list(self.iter_lcs())
