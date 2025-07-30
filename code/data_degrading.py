from os.path import isdir
from os.path import dirname

import numpy as np
import pandas as pd

import data_preparation as dp

CONST = (2 * np.sqrt(2 * np.log(2)))


def degrade_dataframe(R, sn_data, save_path_C=None, save_path_R=None,
                      plot=False):
    if (save_path_C is not None) and (save_path_R is not None):
        if not isdir(dirname(save_path_R)):
            raise FileNotFoundError(f"Directory '{dirname(save_path_R)}' does not exist.")
        if not isdir(dirname(save_path_C)):
            raise FileNotFoundError(f"Directory '{dirname(save_path_C)}' does not exist.")

    # The function below neatly and reproducibly extracts all of the relevant 
    # subsets of the dataframe.
    data = dp.extract_dataframe(sn_data)
    index = data[0]  # SN Name for each spectrum
    wvl0 = data[1]  # Wavelength array
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    df_metadata = data[5]  # Sub-dataframe containing only the metadata
    fluxes0 = data[6]  # Only the flux values in a numpy array

    # Perform degradation (i.e. lowers resolution) for each spectrum in the dataset. The function
    # degraded_spectrum is vectorized, so supplying multiple spectrum allows
    # the operation to be parallelized.
    fluxes_convolve, wvl_degraded, fluxes_degraded = degrade_spectrum(
        R, wvl0, fluxes0)

    # `wvl_degraded` is actually multiple copies of the same wavelength array, 
    # so we only need the first one.
    degraded_flux_columns = wvl_degraded[0].astype(str)

    # Store the convolution data (not the rebinned spectra)
    sn_data_convolve = sn_data.copy(deep=True)
    sn_data_convolve.loc[:, flux0_columns] = fluxes_convolve

    # Store the rebinned spectra into its own dataframe.
    df_fluxes_degraded = pd.DataFrame(data=fluxes_degraded,
                                      columns=degraded_flux_columns,
                                      index=index,
                                      dtype=float)
    sn_data_degraded = pd.concat([df_metadata, df_fluxes_degraded], axis=1)

    if (save_path_C is not None) and (save_path_R is not None):
        sn_data_convolve.to_parquet(save_path_C)
        print(f"Saved: {save_path_C}")
        sn_data_degraded.to_parquet(save_path_R)
        print(f"Saved: {save_path_R}")
    if plot:
        import matplotlib.pylab as plt
        plt.plot(flux0_columns.astype(float),
                 sn_data_convolve.iloc[:1].loc[:,flux0_columns].T,
                 label="convolved")

        plt.plot(degraded_flux_columns.astype(float),
                 sn_data_degraded.iloc[:1].loc[:, degraded_flux_columns].T,
                 label="lower res")
        plt.show()
    return sn_data_convolve, sn_data_degraded


def degrade_spectrum(
    R,
    wvl0,
    flux0,
    wvl_range=(2_500, 10_000),
    ):
    """
    Degrade the spectral resolution of one spectrum (wvl0 and flux0) to R.

    Args:
        R: float
            The desired spectral resolution. Must be less than whatever the
            current spectral resolution of the input spectrum is.
        wvl0: (N,) numpy array
            Array of the wavelength bin centers of the spectrum.
        flux0: (N,) numpy array
            Array of the flux values corresponding to each of the wavelength
            bins of the spectrum.
        wvl_range: (2,) tuple of floats
            The minimum and maximum wavelength values that denote the range of
            wavelengths to rebin the spectrum to. I believe that the output of
            SNID is always a spectrum between 2,500 angstroms and 10,000
            angstroms.

    Returns:
        wvl: (M,) numpy array
            Wavelength bin centers representing the new wavelength bins for
            the degraded spectrum. M (the length of the resulting spectrum)
            should always be less than N (the length of the input spectrum).
        flux: (M,) numpy array
            Flux values corresponding to wvl.
    """
    # Calculate the spectral resolution, R, of the original data.
    R_current = np.mean(wvl0[:-1] / np.diff(wvl0))
    assert R < R_current, "cannot make resolution higher, pass a different R"

    # Get the new wavelength bins defined by R.
    wvl, wvl_bin_sizes = calc_new_wvl(R, wvl0, wvl_range)

    # Need the bin sizes of the existing array which we can calculate in a
    # roundabout way with the calc_new_wvl function.
    tmp_wvl, wvl0_bin_sizes = calc_new_wvl(R_current, wvl0, wvl_range)
    assert np.all(np.isclose(tmp_wvl, wvl0)), "something weird w wavelength bins "

    # As we convolve a Gaussian with the current fluxes, the standard
    # deviation of the Gaussian should be proportional to the wavelength bin
    # sizes with constant of proportionality R_current / R.
    sd_arr = wvl0_bin_sizes * (R_current / R) / CONST

    # Perform the convolution
    flux_conv = special_convolution(wvl0, flux0, wvl0, sd_arr)

    # Interpolate the convolved spectra at the new wavelength centers
    # to get the degraded spectra.
    flux = np.interp(wvl, wvl0, flux_conv)

    return flux_conv, wvl, flux
degrade_spectrum = np.vectorize(degrade_spectrum,
                                signature=("(),(n),(n)->(n),(m),(m)"))


def calc_new_wvl(R, wvl0, wvl_range):
    num_wvl_bins = calc_num_wvl_bins(R, wvl_range)

    d_log_wvl = np.log(wvl_range[1] / wvl_range[0]) / num_wvl_bins
    assert np.isclose(R, d_log_wvl**-1, rtol=0, atol=1), (d_log_wvl**-1, R)

    log_wvl_left_edge = np.arange(num_wvl_bins+1) * d_log_wvl
    log_wvl_left_edge += np.log(wvl_range[0])
    wvl_left_edge = np.exp(log_wvl_left_edge)
    wvl_bin_sizes = np.diff(wvl_left_edge)

    # This code assumes that the center of the bins corresponds to halfway (in
    # linear space, not log space) between the left edge and right edge of the
    # bin.
    wvl = (wvl_left_edge[1:] + wvl_left_edge[:-1]) * 0.5

    new_R = np.mean(wvl / wvl_bin_sizes)
    assert np.isclose(R, new_R, rtol=0, atol=1), "cannot use the R given"

    return wvl, wvl_bin_sizes


def calc_num_wvl_bins(R, wvl_range):
    num_wvl_bins = R * np.log(wvl_range[1] / wvl_range[0])
    num_wvl_bins = int(np.ceil(num_wvl_bins))
    return num_wvl_bins


def special_convolution(wvl0, flux0, mu, sigma):
    weight = gaussian(wvl0, mu, sigma)

    # Renormalize the PSF so that the total integral is 1. This is
    # necessary to handle edge effects.
    renorm = np.trapz(weight, x=wvl0)
    weight /= renorm

    # Perform one step of the integration of t
    flux_conv = np.trapz(flux0 * weight, x=wvl0)

    return flux_conv

special_convolution = np.vectorize(special_convolution,
                                   signature="(n),(n),(),()->()")


def gaussian(x, mu, sigma):
    """
    Evaluates a Gaussian with mean mu and standard deviation sigma.

    We use a gaussian to approximate the PSF of a general spectrograph.

    Args:
        x: array-like
            The points to evaluate the Gaussian. For this work, these will be
            wavelength values.
        mu: float
            The mean of the Gaussian.
        sigma: float
           The standard deviation of the Gaussian.

    Returns:
        gauss : array-like
    """
    normalization = 1 / (np.sqrt(2 * np.pi) * sigma)
    gaussian = np.exp((-1/2) * ((x - mu) / sigma)**2)
    gaussian *= normalization
    return gaussian
