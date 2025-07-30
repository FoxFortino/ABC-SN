# Data

This folder contains all data related to the training of ABC-SN.

```
data/
├─── original_lnw_files/
│    ├─── from_DASH
│    │    └─── lnw files...
│    └─── from_Liu_Modjaz
│         └─── lnw files...
├─── original_resolution_parquet/
│    ├─── original_data.parquet  # All lnw files collated into one DataFrame. lnw_to_parquet.py creates this file.
│    ├─── df_preprocessed.parquet
│    ├─── df_P_tst.parquet
│    ├─── df_P_trn.parquet
│    └─── df_PA_trn.parquet
├─── resolution_100_parquet/
│    ├─── df_PR_tst.parquet
│    ├─── df_SPR_tst.parquet *
│    ├─── df_PAR_trn.parquet
│    ├─── df_SPAR_trn.parquet *
│    ├─── df_PC_tst.parquet
│    └─── df_PAC_trn.parquet
└─── lnw_to_parquet.py
```

The folder `original_lnw_files/` contains the original `.lnw` files collected by Muthukrishna et al. 2019 (in `from_DASH/`) and files collected by Liu and Modjaz (in `from_Liu_Modjaz/`). A README file in `from_Liu_Modjaz/` explains the provenance of that data further. `.lnw` files contain one or more spectra for a given SN. Metadata is also present in these files such as the details of the spline fit to the continuum for each spectrum, the phase of the SN for each spectrum, the set of wavelengths the spectrum is defined on, and the subtype of the SN. These `.lnw` files are all from the SNID templates (Blondin & Tonry 2007).

The folder `original_resolution_parquet/` contains Pandas DataFrame files stored in the `.parquet` format. PyArrow is necessary to read and write these files. `original_data.parquet` contains all of the SN spectra from the `.lnw` files. Some culling of known bad SNe has also been done. The conversion from `.lnw` to Pandas is in `lnw_to_parquet.py`. The known bad SNe that are culled at this stage are listed in `/ABC-SN/code/abcsn_config.py`. `df_preprocessed.parquet` contains the same data but with further culling and standardization applied as detailed in the paper (see main README). Various versions of this data, including after a train-test split as been applied. The naming conventions can be decoded with the table below.

The folder `resolution_100_parquet/` contains the same Pandas DataFrame files but the resolution of all spectra have been lowered (the `R` or `C` in the filenames denote this). The training and testing set files denoted with the `*` represent the data we used to train ABC-SN.

Key for understanding the file naming conventions:

| Filename Element | Meaning |
| ------------ | --- |
| `df`         | A pandas DataFrame |
| `.parquet`   | A parquet file, a more efficient way to store tabular data than `.csv`. Requires PyArrow to be installed for Pandas to read it. |
| `_tst`       | This dataframe contains only testing set spectra. |
| `_trn`       | This dataframe contains only training set spectra. |
| `P`          | 'preprocessed' These spectra have had preprocessing applied including culling and standardzing. |
| `R`          | 'rebinned' The resolution of these spectra have been lowered AND the spectra have been re-binned to the low-res wavelength bins. |
| `C`          | 'continuous' The resolution of these spectra have been lowered AND the spectra have NOT been re-binned so they remain defined on the original high-res wavelength bins. Effectively they are smoothed. |
| `S`          | 'subset' Six SN subtypes have been removed from these datasets.|
| `*`          | This denotes the training and testing set files that were used in the training of ABC-SN. |

