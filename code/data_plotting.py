import numpy as np
from matplotlib import pyplot as plt

import data_preparation as dp


def plot_specs(sn_data, ind, ncols=4, scale=4, xlim=(None, None)):
    # The function below neatly and reproducibly extracts all of the relevant
    # subsets of the dataframe.
    data = dp.extract_dataframe(sn_data)
    index = data[0]  # SN Name for each spectrum
    wvl0 = data[1]  # Wavelength array
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    metadata_columns = data[3]  # Columns that index the metadata
    df_fluxes0 = data[4]  # Sub-dataframe containing only the fluxes
    df_metadata = data[5]  # Sub-dataframe containing only the metadata
    fluxes0 = data[6]  # Only the flux values in a numpy array

    nrows = int(np.ceil(ind.size / ncols))

    fig, ax = plt.subplots(
        nrows, ncols, sharex=False, sharey=False, figsize=(ncols * scale, nrows * scale)
    )
    axes = ax.flatten()

    for i, row in enumerate(sn_data.iloc[ind].iterrows()):
        sn_name = row[0]
        sn_subtype = row[1]["SN Subtype"]
        sn_phase = row[1]["Spectral Phase"]
        title = f"{sn_name} | {sn_subtype} | {sn_phase}"
        axes[i].set_title(title)

        spectrum = row[1][flux0_columns].to_numpy(float)
        axes[i].plot(wvl0, spectrum)

        axes[i].axhline(y=0, c="k", ls=":")
        
        axes[i].set_xlim(xlim)
        
    fig.show()


def plot_loss(log, scale=8, plot_residuals=True, cb_es=None, patience_es=None):

    nrows, ncols = 2, 1
    figsize = (scale * np.sqrt(2), scale)
    if plot_residuals:
        nrows = 3
        figsize = (scale * np.sqrt(2), scale * np.sqrt(2))
        
    
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=False,
        figsize=figsize,
    )
    plt.subplots_adjust(hspace=0)

    # Plot training and testing loss
    axes[0].plot(
        log["epoch"],
        log["loss"],
        c="tab:blue",
        label="Training",
    )
    axes[0].plot(
        log["epoch"],
        log["val_loss"],
        c="tab:orange",
        label="Testing",
    )

    # Plot training and testing categorical accuracy
    axes[1].plot(
        log["epoch"],
        log["ca"],
        c="tab:blue",
        ls="-",
        label="Accuracy (Trn)",
    )
    axes[1].plot(
        log["epoch"],
        log["val_ca"],
        c="tab:orange",
        ls="-",
        label="Accuracy (Tst)",
    )

    # Plot training and testing macro F1-score
    axes[1].plot(
        log["epoch"],
        log["f1"],
        c="tab:blue",
        ls=":",
        label="F1-Score (Trn)",
    )
    axes[1].plot(
        log["epoch"],
        log["val_f1"],
        c="tab:orange",
        ls=":",
        label="F1-Score (Tst)",
    )

    if plot_residuals:
        # Plot residuals between training and testing accuracy
        res_ca = np.array(log["ca"]) - np.array(log["val_ca"])
        axes[2].plot(
            log["epoch"],
            res_ca,
            c="k",
            ls="-",
            label="Accuracy (Trn - Tst)",
        )
        
        # Plot residuals between training and testing F1
        res_f1 = np.array(log["f1"]) - np.array(log["val_f1"])
        axes[2].plot(
            log["epoch"],
            res_f1,
            c="k",
            ls=":",
            label="F1-Score (Trn - Tst)",
        )
        axes[2].set_ylim((-0.05, None))
        axes[2].grid()
        axes[2].set_ylabel("Metric Residuals")
        axes[2].legend(loc="upper left")
        axes[2].axhline(y=0, c="k", lw=2)

    if cb_es is not None:
        annot_x = 1.01
        v_space = 0.2
        v_start = 0.70
        best_epoch = cb_es.stopped_epoch - (patience_es - 1)
        axes[1].annotate(
            f"Loss (Trn):\n    {log['loss'][best_epoch]:.2f}",
            (best_epoch, log['f1'][best_epoch]),
            (annot_x, v_start + 3 * v_space),
            textcoords="axes fraction")
        axes[1].annotate(
            f"Loss (Tst):\n    {log['val_loss'][best_epoch]:.2f}",
            (best_epoch, log['val_f1'][best_epoch]),
            (annot_x, v_start + 2 * v_space),
            textcoords="axes fraction")

        
        axes[1].annotate(
            f"F1-Score (Trn):\n    {log['f1'][best_epoch]*100:.1f}%",
            (best_epoch, log['f1'][best_epoch]),
            (annot_x, v_start),
            textcoords="axes fraction")
        axes[1].annotate(
            f"F1-Score (Tst):\n    {log['val_f1'][best_epoch]*100:.1f}%",
            (best_epoch, log['val_f1'][best_epoch]),
            (annot_x, v_start - v_space),
            textcoords="axes fraction")

        axes[1].annotate(
            f"Accuracy (Trn):\n    {log['ca'][best_epoch]*100:.1f}%",
            (best_epoch, log['f1'][best_epoch]),
            (annot_x, v_start - 3 * v_space),
            textcoords="axes fraction")
        axes[1].annotate(
            f"Accuracy (Tst):\n    {log['val_ca'][best_epoch]*100:.1f}%",
            (best_epoch, log['val_f1'][best_epoch]),
            (annot_x, v_start - 4 * v_space),
            textcoords="axes fraction")
        
        # Plot a vertical line when validation loss is at minimum.
        for i, ax in enumerate(axes):
            if i == 0:
                label = f"Saved model at epoch {best_epoch}"
            else:
                label = ""
            ax.axvline(
                x=best_epoch,
                c="tab:red",
                ls="--",
                label=label,
            )

    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim((0, None))

    axes[0].set_ylabel("Categorical Cross Entropy")
    axes[1].set_ylabel("Metric")

    # axes[0].set_ylim((None, None))
    axes[1].set_ylim((0, 0.99))

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper left")

    axes[0].grid()
    axes[1].grid()

    axes[0].set_yscale("log")
    # axes[1].set_yscale("log")

    return fig


def plot_1D_results(mg, keys, vals, scale=8):
    retry_axis = keys.index("retry")
    del keys[retry_axis]
    del vals[retry_axis]

    params = vals[0]
    param_name = keys[0]

    fig, axes = plt.subplots(
        3, 1, sharex=True, sharey=False, figsize=(scale, scale * np.sqrt(2))
    )
    fig.subplots_adjust(hspace=0, wspace=0)

    for metric in range(mg.shape[-1]):
        if metric == 0:
            title = "Categorical Cross-Entropy"
        if metric == 1:
            title = "Categorical Accuracy"
        if metric == 2:
            title = "F1-Score"

        ax = axes[metric]

        ax.set_ylabel(title)

        mu = np.nanmean(mg[..., 0, metric], axis=retry_axis)
        sd = np.nanstd(mg[..., 0, metric], axis=retry_axis)
        ax.errorbar(
            params,
            mu,
            yerr=sd,
            c="tab:blue",
            marker="o",
            ls="--",
            capsize=5,
            elinewidth=1,
            label=f"Train\nMax: {mu.max():.3f}\nMin: {mu.min():.3f}",
        )

        mu = np.nanmean(mg[..., 1, metric], axis=retry_axis)
        sd = np.nanstd(mg[..., 1, metric], axis=retry_axis)
        ax.errorbar(
            params,
            mu,
            yerr=sd,
            c="tab:orange",
            marker="o",
            ls="--",
            capsize=5,
            elinewidth=1,
            label=f"Test\nMax: {mu.max():.3f}\nMin: {mu.min():.3f}",
        )

        ax.legend(loc="center left")

    axes[-1].set_xlabel(param_name)

    return fig


def plot_random_spectra(df, N, ncols=4, scale=4):
    inds = np.random.randint(low=0, high=df.shape[0], size=N)

    nrows = int(np.ceil(inds.size / ncols))

    fig, ax = plt.subplots(
        nrows, ncols, sharex=False, sharey=False, figsize=(ncols * scale, nrows * scale)
    )
    axes = ax.flatten()

    for i, row in enumerate(df.iloc[inds].iterrows()):
        sn_name = row[0]
        sn_subtype = row[1]["SN Subtype"]
        sn_phase = row[1]["Spectral Phase"]
        title = f"{sn_name} | {sn_subtype} | {sn_phase}"
        axes[i].set_title(title)

        spectrum = row[1][flux_columns].to_numpy(float)
        axes[i].plot(wvl0, spectrum)

        axes[i].axhline(y=0, c="k", ls=":")
    fig.show()


@np.vectorize
def wavelen2rgb(nm):
    """
    Converts a wavelength between 380 and 780 nm to an RGB color tuple.

    Willow: This code taken from rsmith-nl/wavelength_to_rgb git repo.

    Arguments
    ---------
        nm : float
            Wavelength in nanometers.
    Returns
    -------
        rgb : 3-tuple
            tuple (red, green, blue) of integers in the range 0-255.
    """

    def adjust(color, factor):
        if color < 0.01:
            return 0
        max_intensity = 255
        gamma = 0.80
        rv = int(round(max_intensity * (color * factor) ** gamma))
        if rv < 0:
            return 0
        if rv > max_intensity:
            return max_intensity
        return rv

    # if nm < 380 or nm > 780:
    #     raise ValueError('wavelength out of range')
    if nm < 380:
        nm = 380
    if nm > 780:
        nm = 780

    red = 0.0
    green = 0.0
    blue = 0.0
    # Calculate intensities in the different wavelength bands.
    if nm < 440:
        red = -(nm - 440.0) / (440.0 - 380.0)
        blue = 1.0
    elif nm < 490:
        green = (nm - 440.0) / (490.0 - 440.0)
        blue = 1.0
    elif nm < 510:
        green = 1.0
        blue = -(nm - 510.0) / (510.0 - 490.0)
    elif nm < 580:
        red = (nm - 510.0) / (580.0 - 510.0)
        green = 1.0
    elif nm < 645:
        red = 1.0
        green = -(nm - 645.0) / (645.0 - 580.0)
    else:
        red = 1.0
    # Let the intensity fall off near the vision limits.
    if nm < 420:
        factor = 0.3 + 0.7 * (nm - 380.0) / (420.0 - 380.0)
    elif nm < 701:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780.0 - nm) / (780.0 - 700.0)
    # Return the calculated values in an (R,G,B) tuple.
    return (adjust(red, factor), adjust(green, factor), adjust(blue, factor))


def plot_spec(wvl, flux, scale=8, err=None, save=None):
    """
    Plot a spectrum with appropriate colors.

    Arguments
    ---------
    wvl : (N,) array-like
        Array defining the wavelength bin centers of the spectrograph.
    flux : (N,) array-like
        Array of flux values for each wavelength bin.

    Keyword Arguments
    -----------------
    err : (N,) array-like, Default: None
        Array of uncertainties in the flux measurement. If None, no errorbars
        are plotted.
    save : str, Default: None
        If save is None, then the resulting plot is not saved. If save is a
        string, then plt.savefig will be called with that string.
    """

    if not np.any(wvl > 7000):
        RGB = wavelen2rgb(wvl / 10)
        RGBA = np.array(RGB).T / 255
    else:
        # If there are wavelength points above 7000 angstroms, make them an
        # RGB value corresponding to 7000 angstroms. This RGB code can't
        # handle wavelengths not between 4000 and 7000 angstroms.
        over7000 = np.where(wvl > 7000)[0]
        wvl_copy = wvl.copy()
        wvl_copy[over7000] = 7000
        RGB = wavelen2rgb(wvl_copy / 10)
        RGBA = np.array(RGB).T / 255

    errmsg = (
        "flux and wvl arrays should be of same size but are "
        f"{flux.size} and {wvl.size} respectively. Each flux value "
        "should correspond to one wavelength bin. See docstring for "
        "more info."
    )
    assert wvl.size == flux.size, errmsg

    wvl_LE = adjust_logbins(wvl)

    fig, ax = plt.subplots(figsize=(2 * scale, 1 * scale))

    if err is not None:
        ax.errorbar(
            wvl[:-1],
            flux[:-1],
            yerr=err[:-1],
            elinewidth=1,
            capsize=3,
            ls="-",
            c="k",
            marker="*",
        )
    else:
        ax.plot(wvl[:-1], flux[:-1], ls="", c="k", marker="")
    _, _, patches = ax.hist(wvl_LE[:-1], bins=wvl_LE, weights=flux[:-1], align="mid")

    # Each patch of the histogram (the rectangle underneath each point) gets
    # colored according to its central wavelength.
    for patch, color in zip(patches, RGBA):
        patch.set_facecolor(color)

    ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=40)
    ax.set_ylabel(r"Normalized Flux [$F_{\lambda}$]", fontsize=40)

    # bounds = flux.max() + flux.max()*0.05
    # ax.ylim((-bounds, bounds))

    ax.set_ylim((0, 1.05))

    ax.set_ylim((flux.min(), flux.max()))
    ax.set_xlim((3800, 7000))

    ax.tick_params(axis="both", which="major", labelsize=30)
    fig.tight_layout()

    if save is not None:
        ax.savefig(save)

    return fig, ax


def adjust_logbins(bins, current="center", new="leftedge"):
    """
    Redefines whether an array corresponds to bin centers or bin edges.

    Assuming the bins have a constant spacing in log-space (that is:
    ``np.diff(np.log(bins))`` is a constant array) then this function will
    shift the bins from bin-center-defined to bin-edge-defined or vice versa.

    Arguments
    ---------
    bins : array-like
        One dimensional array of bin positions.
    current : {"center", "leftedge",}, default: "center"
        Whether the bins array currently defines the bin centers or left edge.
    new : {"center", "leftedge"}, default: "leftedge"
        What the returned bins array should define: the bin centers or the left
        bin edge.

    Returns
    -------
    new_bins : array-like
        One dimensional array of new bin positions.
    """

    logbin = np.log(bins)
    d_logbin = np.mean(np.diff(logbin))

    if current == "center" and new == "leftedge":
        # diff_logbin / 2 will give the bin radius in log space. If our array
        # denotes bin-centers, then we need to subtract the bin radius from
        # the array so that now the array is denoting left-bin-edges.
        # Also note we need to add one more bin in log space before we take
        # np.diff. This is so that when we subtract arrays of the same shape
        # in the next line.
        bin_radii = np.diff(logbin, append=logbin[-1] + d_logbin)
        new_logbins = logbin - bin_radii * 0.5
        new_bins = np.exp(new_logbins)

    elif current == "leftedge" and new == "center":
        bin_widths = np.diff(logbin, append=logbin[-1] + d_logbin)
        new_logbins = logbin + bin_widths * 0.5
        new_bins = np.exp(new_logbins)

    return new_bins


def plot_cm(cm, classes, R, figsize=(10, 10), title=False):
    textargs = {"fontname": "Serif"}
    
    # Normalize confusion matrix and set image parameters
    cm = cm.astype("float") / np.nansum(cm, axis=1)[:, np.newaxis]
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1

    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation="none", cmap="RdBu", vmin=-1, vmax=1)
    
    cbticks = np.linspace(-1, 1, num=9)
    cbticklabels = ["100%", "75%", "50%", "25%", "0%", "25%", "50%", "75%", "100%"]
    cb = plt.colorbar(im, shrink=0.82)
    cb.set_ticks(cbticks, labels=cbticklabels, fontsize= 12, **textargs)
    
    if title:
        ax.set_title(title, **textargs, fontsize=15)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=90, **textargs, fontsize= 14)
    ax.set_yticks(tick_marks, classes, **textargs, fontsize= 14)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = np.abs(cm[i, j])
            if val == 0:
                text = ""
            elif val == 1:
                text = "100"
            else:
                text = f"{val*100:.1f}"
            color = "w" if val >= 0.50 else "k"
            ax.text(
                j, i, text,
                ha="center", va="center",
                c=color, **textargs, fontsize=11,
            )

    ax.set_ylabel("True label", fontsize=20, **textargs)
    ax.set_xlabel("Predicted label", fontsize=20, **textargs)
    
    return fig
