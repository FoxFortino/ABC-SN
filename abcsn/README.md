# ABC-SN Model Files

The `.keras` model file for the final trained model is too large to be hosted on GitHub. Find it on Zenodo [here]([url](https://zenodo.org/records/16620817)). When you download it from Zenodo yourself, place it in this directory and ensure it is named `ABCSN.keras`.

`ABCSN_pretrain.keras` — The model used to pretrain the weights in the transformer encoder layers of ABC-SN.

`fig00_pretrain_masked_spectra_reconstruction.png` — Four random spectra from the test set and how they are recreated by the pretraining model.

`fig01_pretrain_loss.png` — Loss curve for the pretraining model.

`fig02_transfer_loss.png` - Loss and performance curves for ABC-SN.

`fig03_transfer_CMtrn.png` — Confusion matrix on the training set for ABC-SN, normalized by row (_i.e._, for recall/completeness).

`fig04_transfer_CMtst.png` — Confusion matrix on the test set for ABC-SN, normalized by row (_i.e._, for recall/completeness).

`pretrain_log.csv` — Learning rate, training set loss and test set loss during training of the pretraining model.

`transfer_log.csv` — Learning rate, categorical accuracy (ca), macro F1-score (f1), categorical cross-entropy loss (loss) on the training and test set during the training of ABC-SN.

`train_abcsn.sh` — The SLURM job script that was used to start the training of the pretraining model and ABC-SN. This file is likely not useful to anyone and is included for completeness of records associated with ABC-SN.

`slurm-5566096.out` — The full output file for the training of the pretraining model and ABC-SN.
