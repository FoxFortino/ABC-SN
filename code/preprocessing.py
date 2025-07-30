from os.path import join

import numpy as np
import pandas as pd
from scipy import stats

import data_preparation as dp
import data_degrading as dg

rng = np.random.RandomState(1415)


def preprocessing(
    R=100,
    df_raw="../data/original_resolution_parquet/original_data.parquet",
    save_dir_original_R="../data/original_resolution_parquet/",
    save_dir_R100="../data/resolution_100_parquet/",
    phase_range=(-20, 50),
    ptp_range=(0.1, 100),
    wvl_range=(4500, 7000),
    train_frac=0.50,
    rng=rng,
):
    """
    All preprocessing steps for ABC-SN.

    This script should not have to be rerun since the all of the outputs are saved in the data folder.
    """
    print(f"Loading spectra: '{df_raw}'")
    df_raw = pd.read_parquet(df_raw)
    print(f"Shape of `df_raw`: {df_raw.shape}")

    print(f"Preliminary precoessing of spectra:")
    print(f"    Remove spectra with a phase outside of the range {phase_range} days.")
    print(f"    Removing spectra with a peak-to-peak ratio outside of the range {ptp_range}.")
    print(f"    Setting all flux values to zero at wavelengths outside of the range {wvl_range} angstroms.")
    print(f"    Standardizing the non-zero portion of each spectrum.")
    df_prep = preproccess_dataframe(df_raw, phase_range, ptp_range, wvl_range)
    df_prep.to_parquet(join(save_dir_original_R, "df_preprocessed.parquet"))
    print(f"Shape of `df_prep`: {df_prep.shape}")

    print(f"Performing the train-test split on the preprocessed dataset.")
    print(f"    Fraction of the supernovae to be in the training set: {train_frac}.")
    df_P_trn, df_P_tst = split_data(df_prep, train_frac, rng)    
    df_P_trn.to_parquet(join(save_dir_original_R, "df_P_trn.parquet"))
    df_P_tst.to_parquet(join(save_dir_original_R, "df_P_tst.parquet"))
    print(f"Shape of `df_P_trn`: {df_P_trn.shape}")
    print(f"Shape of `df_P_tst`: {df_P_tst.shape}")

    print(f"Performing data augmentation on the preprocessed training set.")
    df_PA_trn = augment(df_P_trn, wvl_range)
    df_PA_trn.to_parquet(join(save_dir_original_R, "df_PA_trn.parquet"))
    print(f"Shape of `df_PA_trn`: {df_PA_tst.shape}")

    print(f"Lowering spectral resolution of training set spectra to R = {R}...")
    df_PAC_trn, df_PAR_trn = dg.degrade_dataframe(R, df_PA_trn)
    df_PAC_trn.to_parquet(join(save_dir_R100, "df_PAC_trn.parquet"))
    df_PAR_trn.to_parquet(join(save_dir_R100, "df_PAR_trn.parquet"))
    print(f"Shape of `df_PAC_trn`: {df_PAC_trn.shape}")
    print(f"Shape of `df_PAR_trn`: {df_PAR_trn.shape}")
    print(f"The 'C' in 'df_PAC_trn' denotes that these spectra are defined on the same wavelengths as the original spectra.")
    print(f"The 'R' in 'df_PAR_trn' denotes that these spectra are defined on the re-binned wavelengths at R = {R}.")

    print(f"Lowering spectral resolution of testing set spectra to R = {R}...")
    df_PC_tst, df_PR_tst = dg.degrade_dataframe(R, df_P_tst)
    df_PC_tst.to_parquet(join(save_dir_R100, "df_PC_tst.parquet")
    df_PR_tst.to_parquet(join(save_dir_R100, "df_PR_tst.parquet")
    print(f"Shape of `df_PC_tst`: {df_PC_tst.shape}")
    print(f"Shape of `df_PR_tst`: {df_PR_tst.shape}")
    print(f"The 'C' in 'df_PC_tst' denotes that these spectra are defined on the same wavelengths as the original spectra.")
    print(f"The 'R' in 'df_PR_tst' denotes that these spectra are defined on the re-binned wavelengths at R = {R}.")
    return


def preproccess_dataframe(sn_data, phase_range, ptp_range, wvl_range):
    # The function below neatly and reproducibly extracts all of the relevant 
    # subsets of the dataframe.
    data = dp.extract_dataframe(sn_data)
    wvl0 = data[1]  # Wavelength array
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    fluxes0 = data[6]  # Only the flux values in a numpy array

    # Spectra with a spectral phase outside of `phase_range`.
    bad_ind = sn_data["Spectral Phase"] < phase_range[0]
    bad_ind |= sn_data["Spectral Phase"] > phase_range[1]

    # Remove the spectra with a peak to valley that is too small or too large.
    ptp = np.ptp(fluxes0, axis=1)
    bad_ind |= ptp < ptp_range[0]
    bad_ind |= ptp > ptp_range[1]
    
    # Remove spectra that are completely flat within `wvl_range` (all the same value) for some reason.
    wvl_range_mask = (wvl0 < wvl_range[0]) | (wvl0 > wvl_range[1])
    flat_spectra_inds = np.std(fluxes0, axis=1, where=~wvl_range_mask) == 0
    bad_ind |= flat_spectra_inds

    # `bad_ind` now is True for all rows that we want to remove, so now we set `fluxes0` to only the rows we want.
    fluxes0 = fluxes0[~bad_ind]
    
    standardized_fluxes0 = standardize_fluxes(fluxes0, wvl0, wvl_range)

    # Set the standardized flux data into the dataframe using `~bad_ind` to index only the rows that were not removed.
    sn_data.loc[~bad_ind, flux0_columns] = standardized_fluxes0

    # Remove the rows that we have pruned above.
    sn_data = sn_data.loc[~bad_ind]

    return sn_data


def standardize_fluxes(fluxes, wvl, wvl_range):
    # Use `wvl_range` to define a mask that is True outside of `wvl_range` and False inside.
    wvl_range_mask = (wvl < wvl_range[0]) | (wvl > wvl_range[1])

    # Standardize the dataset to zero mean and standard deviation of 1.
    flux_means = np.mean(fluxes, axis=1, where=~wvl_range_mask)[..., None]
    flux_stds = np.std(fluxes, axis=1, where=~wvl_range_mask)[..., None]
    standardized_fluxes = (fluxes - flux_means) / flux_stds

    # Set all flux values outside of `wvl_range` to 0.
    standardized_fluxes[:, wvl_range_mask] = 0
    
    # Check that the standardization worked.
    assert np.all(standardized_fluxes[:, wvl_range_mask] == 0), "All data points outside of `wvl_range` should be 0."
    
    mean_inside_wvl_range = np.mean(standardized_fluxes, axis=1, where=(~wvl_range_mask))
    assert np.all(np.isclose(mean_inside_wvl_range, 0)), "Mean of all data points inside `wvl_range` should be 0."
    
    stdv_inside_wvl_range = np.std(standardized_fluxes, axis=1, where=(~wvl_range_mask))
    assert np.all(np.isclose(stdv_inside_wvl_range, 1)), "Stddev of all data points inside `wvl_range` should be 1."
    
    return standardized_fluxes


def df_split(x, train_frac, rng):
    x["Exclude"] = False
    x["Training Set"] = False

    sn_names = x.index.unique().to_list()
    num_supernova = len(sn_names)
    if num_supernova == 1:
        x["Exclude"] = True
        return x

    num_train = int(np.ceil(num_supernova * train_frac))
    if num_supernova - num_train == 0:
        num_train -= 1

    inds = rng.choice(sn_names,
                      size=num_train,
                      replace=False)
    x.loc[inds, "Training Set"] = True
    return x


def split_data(sn_data, train_frac, rng):

    sn_data_split = sn_data.groupby(by=["SN Subtype"],
                                    axis=0,
                                    group_keys=True).apply(df_split,
                                                           train_frac,
                                                           rng)
    training_set = sn_data_split["Training Set"] & ~sn_data_split["Exclude"]
    testing_set = ~sn_data_split["Training Set"] & ~sn_data_split["Exclude"]
    sn_data_trn = sn_data_split.loc[training_set]
    sn_data_tst = sn_data_split.loc[testing_set]

    sn_data_trn.reset_index(level="SN Subtype", drop=True, inplace=True)
    sn_data_tst.reset_index(level="SN Subtype", drop=True, inplace=True)
    return sn_data_trn, sn_data_tst


def gen_noise(spectrum, rng):
    """Generate Gaussian noise to add to spectra with mean zero and standard deviation of 0.10."""
    noise = stats.norm.rvs(loc=0, scale=0.10, size=spectrum.size, random_state=rng)
    return noise
gen_noise = np.vectorize(gen_noise, signature="(n),()->(n)")


def gen_redshift(spectrum, rng):
    """Loosely simulate redshift estimation error by shifting the entire spectrum by at most 5 pixels left or right."""
    shift_amount = stats.randint.rvs(-5, 6, size=1, random_state=rng).item()
    shifted_spectrum = np.roll(spectrum, shift_amount)
    return shifted_spectrum
gen_redshift = np.vectorize(gen_redshift, signature="(n),()->(n)")


def gen_spikes(spectrum, rng):
    """Loosely simulate telluric lines by adding in one-pixel wide spikes to the dataset."""
    
    # First decide how many spikes should be added. At most 5 and at minimum none.
    num_spikes = stats.randint.rvs(low=0, high=5, size=1, random_state=rng).item()

    # Next decide the location of the spikes by choosing a random number between 0 and the length of the array containing the spectrum. Because `spectrum` is not masked to remove the padding (i.e., it will have 0 padding up to 4500 angstroms and beyond 7000 angstroms), some of these spikes will be placed in the region of the zero-padding which will eventually be overwritten by a 0 in the data augmentation code that this function gets called in. This is fine.
    # Physically, telluric lines should occur in redder wavelengths, but this is not modelled here for simplicity.
    spike_loc = stats.randint.rvs(low=0, high=spectrum.size, size=num_spikes, random_state=rng)

    # Next, we decide if the spike will be an addition or subtraction (i.e., if the telluric line is emission or absorption). 80% of the time, the spike will be in emission (a positive spike) and the rest of the time it will be in absorption.
    spike_dir = stats.binom.rvs(n=1, p=0.80, size=num_spikes, random_state=rng)
    spike_dir[spike_dir == 0] = -1

    # Finally choose the magnitude of the spike. We take the absolute value since we are having the sign of the spike determined by the previous set of code.
    # The standard deviations of the magnitude of the injected spikes will be tied to the standard deviation of the spectrum.
    spike_mag = np.abs(stats.norm.rvs(loc=0, scale=2*spectrum.std(), size=num_spikes, random_state=rng))

    # Construct the array of spikes that is the same shape as the original spectrum.
    spikes = np.zeros_like(spectrum)
    spikes[spike_loc] = spike_mag * spike_dir

    return spikes
gen_spikes = np.vectorize(gen_spikes, signature="(n),()->(n)")


def augment(sn_data, wvl_range):
    # Unpack the dataframe. The dataframe is quite dense, information-wise, so this function unpacks the various pieces of information in a consistent way.
    data = dp.extract_dataframe(sn_data)
    wvl0 = data[1]  # The wavelength array of the spectra
    fluxes = data[6]  # The fluxes as a numpy array.

    # Generate a mask for the spectrum which is False within the wavelength range specified by `wvl_range`, and True otherwise. This way, we can use this mask after the data augmentation steps to make sure that all of the fluxes outside of `wvl_range` are 0.
    wvl_range_mask = (wvl0 < wvl_range[0]) | (wvl0 > wvl_range[1])
    
    # Figure out how many times each spectrum should be repeated when augmenting the dataset.
    sn_types, num_spectra = np.unique(sn_data["SN Subtype"], return_counts=True)
    num_augments = (np.max(num_spectra) - num_spectra) / num_spectra
    num_augments = np.ceil(num_augments).astype(int) + 1
    
    ic(sn_types)
    ic(num_augments)
    
    # Loop through each supernova type, performing the augmentation steps.
    sn_type_df_list = []
    for sn_type, num_augment in zip(sn_types, num_augments):
        # Grab the subset of the dataframe which includes only the rows corresponding to `sn_type`. Call it `sn_type_df`.
        # We first make a copy of the original dataframe so that there is no chance of overwriting the original data with pointers/views.
        df_copy = sn_data.copy(deep=True)
        sn_type_mask = sn_data["SN Subtype"] == sn_type
        sn_type_df = df_copy[sn_type_mask]

        # This generates a dataframe which repeats all of the data in `sn_type_df` the number of times specified by `num_augment`. This repeated data forms the basis for the dataset augmentation.
        sn_type_df_rep = sn_type_df.iloc[np.tile(np.arange(sn_type_df.shape[0]), num_augment)].copy(deep=True)

        # Unpack the dataframe to grab the fluxes and the `flux_columns` which will be used to index the dataframe later on. Note that `flux_columns` is the exact same as the wavelength array but are strings.
        data = dp.extract_dataframe(sn_type_df_rep)
        flux_columns = data[2]
        fluxes = data[6]
        
        # Generate redshifted dataset
        shifted = gen_redshift(fluxes, rng)
        
        # Generate noise to be added to the data.
        noise = gen_noise(fluxes, rng)
        
        # Generate spikes to be added to the data.
        spikes = gen_spikes(fluxes, rng)
        
        # Construct the augmented dataset by replacing fluxes with the shifted fluxes, adding noise and spikes.
        augmented_fluxes = shifted + noise + spikes
        
        # Reset the fluxes outside of `wvl_range` to 0.
        augmented_fluxes[:, wvl_range_mask] = 0
        
        # Put these augmented fluxes back into `sn_type_df_rep` which up until this point still only contained the original fluxes without augmentation.
        sn_type_df_rep.loc[:, flux_columns] = augmented_fluxes
        
        # Appending `sn_type_df_rep` to a list allows us to later use `pd.concat` to combine all of them together after the for loop is complete, creating the final augmented dataset in one simple line of code.
        sn_type_df_list.append(sn_type_df_rep)
        
    # Finally, we have the augmented dataset.
    sn_data_augmented = pd.concat(sn_type_df_list, axis=0)
        
    # Lastly we must re-standardize the data such that each spectrum has mean 0 and standard deviation 1.
    data = dp.extract_dataframe(sn_data_augmented)
    fluxes_aug_columns = data[2]
    fluxes_aug = data[6]
    
    standardized_fluxes_aug = standardize_fluxes(fluxes_aug, wvl0, wvl_range)

    # Set the data in the dataframe to be the re-standardized data.
    sn_data_augmented.loc[:, fluxes_aug_columns] = standardized_fluxes_aug

    return sn_data_augmented

