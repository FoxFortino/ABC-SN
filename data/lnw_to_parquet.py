"""
This script will generate a pandas dataframe from two lists of lnw files.

This script is designed to be run from the command line. You can modify the
hardcoded data directories and save directories. The defaults were appropriate
for my file system but won't be for you. This script only needs to be run
once.

However, the result of this file `sn_data.parquet` should already exist in the
data directory of this package.
"""
import os
from os import getcwd
from os.path import isdir
from os.path import isfile
from os.path import basename
from os.path import splitext
from os.path import join
from os.path import abspath
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(0, "../code")
import abcsn_config


def main(save_dir, data1_lnws, data2_lnws):
    no_duplicate_lnw = remove_duplicate_lnw(data1_lnws, data2_lnws)
    no_bad_lnw = remove_bad_sn(no_duplicate_lnw)
    sn_data0 = load_lnws(no_bad_lnw, save_dir)

    sn_data_file = join(save_dir, "sn_data.parquet")
    print(f"Dataframe saved to: {sn_data_file}")


def remove_duplicate_lnw(files_DASH: list, files_SESN: list):
    # Loop through all files to look for duplicates between the SESN data and
    # the astrodash data. If there is a duplicate, keep the SNID one in the
    # list for no particular reason other than it is probably more updated.
    lnw_SESN = [basename(file) for file in files_SESN]

    files_all = files_DASH + files_SESN
    lnw_all = [basename(file) for file in files_all]

    no_duplicate_lnw = []
    for lnw, file in zip(lnw_all, files_all):
        if file in files_SESN:
            no_duplicate_lnw.append(file)

        elif file in files_DASH:
            if lnw in lnw_SESN:
                continue
            else:
                no_duplicate_lnw.append(file)

    print(
        "Total number of SN .lnw files without duplicates: "
        f"{len(no_duplicate_lnw)}"
    )
    return no_duplicate_lnw


def remove_bad_sn(no_duplicate_lnw):
    no_bad_lnw = []
    for lnw in no_duplicate_lnw:
        if splitext(basename(lnw))[0] not in abcsn_config.ALL_BAD_SN:
            no_bad_lnw.append(lnw)

    print(
        "Total number of SN .lnw files without duplicates or bad SN: "
        f"{len(no_bad_lnw)}"
    )
    return no_bad_lnw


def load_lnws(
    lnw_files: list[str],
    save_dir: str = None,
):
    if (save_dir is not None) and (not isdir(save_dir)):
        raise FileNotFoundError(f"'{save_dir}' directory does not exist.")

    list_header = []
    list_wvl0 = []
    list_fluxes = []
    list_phases = []
    list_sn_Stype_ids = []
    list_sn_Stypes = []
    list_sn_names = []
    list_sn_Mtype_ids = []
    list_sn_Mtypes = []

    for lnw_file in lnw_files:
        loaded_data = load_SNID_lnw(lnw_file)
        list_header.append(loaded_data[0])
        list_wvl0.append(loaded_data[1])
        list_fluxes.append(loaded_data[2])
        list_phases.append(loaded_data[3])
        list_sn_Stype_ids.append(loaded_data[4])
        list_sn_Stypes.append(loaded_data[5])
        list_sn_names.append(loaded_data[6])

        Mtype_ids = abcsn_config.determine_ID_Stype_to_Mtype(loaded_data[4])
        Mtypes = abcsn_config.get_Mtype_str_from_ID(Mtype_ids)

        list_sn_Mtype_ids.append(Mtype_ids)
        list_sn_Mtypes.append(Mtypes)

        assert np.all(np.isclose(list_wvl0, loaded_data[1]))

    arr_fluxes = np.concatenate(list_fluxes, axis=0, dtype=float)
    arr_phases = np.concatenate(list_phases, axis=0, dtype=float)
    arr_sn_Stype_ids = np.concatenate(list_sn_Stype_ids, axis=0, dtype=int)
    arr_sn_Stypes = np.concatenate(list_sn_Stypes, axis=0)
    arr_sn_names = np.concatenate(list_sn_names, axis=0)
    arr_sn_Mtype_ids = np.concatenate(list_sn_Mtype_ids, axis=0, dtype=int)
    arr_sn_Mtypes = np.concatenate(list_sn_Mtypes, axis=0)

    df_sn_names = pd.Series(
        data=arr_sn_names, dtype="category", name="SN Name"
    )
    df_sn_Stypes = pd.Series(
        data=arr_sn_Stypes, dtype="category", name="SN Subtype"
    )
    df_sn_Stype_ids = pd.Series(
        data=arr_sn_Stype_ids, dtype="category", name="SN Subtype ID"
    )
    df_sn_Mtypes = pd.Series(
        data=arr_sn_Mtypes, dtype="category", name="SN Maintype"
    )
    df_sn_Mtype_ids = pd.Series(
        data=arr_sn_Mtype_ids, dtype="category", name="SN Maintype ID"
    )
    df_phases = pd.Series(
        data=arr_phases, dtype="float", name="Spectral Phase"
    )
    df_fluxes = pd.DataFrame(
        data=arr_fluxes, columns=list_wvl0[0].astype(str), dtype=float
    )

    df_data = pd.concat(
        [
            df_sn_Stypes,
            df_sn_Stype_ids,
            df_sn_Mtypes,
            df_sn_Mtype_ids,
            df_phases,
            df_fluxes,
        ],
        axis=1,
    )
    df_data.index = df_sn_names

    if save_dir is not None:
        save_path = join(save_dir, "sn_data.parquet")
        df_data.to_parquet(save_path)

    return df_data


def load_SNID_lnw(lnw_file):
    """
    An adapted version of loadSNIDlnw from SESNspectraPCA.

    Loads the spectral data and metadata from a .lnw SNID template file
    specified by lnw. This function does not extract the continuum information
    from the file.

    Args:
        lnw_file: str
            The path to SNID template file produced by logwave.

    Returns:
        header: dict
            Dictionary containing the original header information contained in
            the lnw file.
        wvl: (N,) numpy array
            Wavelength bin centers. Same for all spectra in the file.
        fluxes: (N, M) numpy array
            Flux values for M different spectra corresponding to M wavelength
            bin centers.
        phases: (M,) numpy array
            The phase of each spectrum in the lnw file.
        sn_type_ids: (M,) numpy array of int
            The supernova type for each spectrum (they will all be the same
            for a given lnw file) as an integer. sn_config.py determines which
            integer corresponds to which supernova type.
        sn_types: (M,) numpy array of str
            The supernova type in English for each spectrum (they will all be
            the same for a given lnw file).
        sn_names: (M,) numpy array of str
            The name of the supernova that this file represents. This array
            exists to correspond each spectrum with its supernova name.
    """
    with open(lnw_file) as lnw:
        lines = lnw.readlines()
        lnw.close()

    # Extract metadata
    header = lines[0].strip().split()
    header = {
        "num_spectra": int(header[0]),
        "num_wvl_bins": int(header[1]),
        "wvl_range_start": float(header[2]),
        "wvl_range_end": float(header[3]),
        "spline_knots": int(header[4]),
        "sn_name": str(header[5]),
        "dm15": float(header[6]),
        "sn_type": str(header[7]),
    }

    # Determine which of the 17 recognized supernova classes is sn_type. First
    # get the integer ID that represents this supernova type. Then convert
    # that back to the name for that type. It may be useful to have both.
    sn_type_id = abcsn_config.SN_Stypes_str_to_int[header["sn_type"]]
    sn_type = abcsn_config.SN_Stypes_int_to_str[sn_type_id]

    # Extract the phase of each spectra in this file.
    phases_line_number = len(lines) - header["num_wvl_bins"] - 1
    phases = lines[phases_line_number].strip().split()

    # In this list comprehension we discrard the first entry which seems to
    # always be 0
    phases = np.array([float(phase) for phase in phases[1:]])

    # Generate a list which is the same length as phases which redundantly
    # denotes the supernova type for each of the spectra in this lnw file.
    # They will all be the same, but because the basic unit of our dataset is
    # the spectrum (we are ultimately trying to classifying supernovae based
    # on their spectrum) we must identify the supernova type with each
    # spectrum, not just with each supernova.
    sn_type_ids = np.full(phases.shape, sn_type_id, dtype=int)
    sn_types = np.full(phases.shape, sn_type)
    sn_names = np.full(phases.shape, header["sn_name"])

    # Load the data at the bottom of the file and extract the wavelength array
    # and the flux data for each spectra corresponding to each phase.
    data = np.loadtxt(lnw_file, skiprows=phases_line_number + 1)
    wvl = np.take(data, 0, axis=1)

    # We take the transpose of this resulting matrix so that each row
    # corresponds to a different spectrum.
    fluxes = np.delete(data, 0, axis=1).T

    return header, wvl, fluxes, phases, sn_type_ids, sn_types, sn_names


def load_sn_data(sn_data_file: os.PathLike) -> pd.DataFrame:
    if not isfile(sn_data_file):
        raise FileNotFoundError(f"File '{sn_data_file}' does not exist.")
    sn_data = pd.read_parquet(sn_data_file)
    return sn_data


if __name__ == "__main__":
    DASH_data_dir = "./original_lnw_files/from_DASH"
    DASH_data_lnws = glob(join(DASH_data_dir, "*lnw"))

    SESN_data_dir = "./original_lnw_files/from_Liu_Modjaz"
    SESN_data_lnws = glob(join(SESN_data_dir, "templates_*/*lnw"))

    save_dir = abspath(getcwd())

    main(save_dir, DASH_data_lnws, SESN_data_lnws)
