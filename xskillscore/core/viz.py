import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xskillscore.core.probabilistic import rank_histogram


def plot_rank_histogram(rank_hist, normalize=True, title=None):
    """
    Plot a Rank Histogram (Talagrand Diagram) from the output of
    xskillscore.rank_histogram().

    This function visualizes the distribution of observation ranks among ensemble members.
    For multi-dimensional rank histograms, all non-'rank' dimensions are summed over,
    resulting in a 1D histogram along the 'rank' axis.

    A uniform rank histogram indicates a well-calibrated ensemble, while deviations
    from uniformity suggest biases or under/over-dispersion in the ensemble forecasts.

    Parameters
    ----------
    rank_hist : xr.DataArray or xr.Dataset
        Output of :py:func:`xskillscore.rank_histogram`.
        Must include a 'rank' dimension. If multi-dimensional, all non-'rank'
        dimensions are collapsed (summed) to produce a 1D histogram.
    normalize : bool, default=True
        If True, converts counts to frequencies (so the histogram sums to 1).
    title : str or None, optional
        Optional custom plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.


    Notes
    -----
    - For multi-dimensional input, all non-'rank' dimensions are summed over.
    - A uniform histogram indicates a well-calibrated ensemble; U- or dome-shaped
      histograms suggest under- or over-dispersion, respectively.

    Examples
    --------
    >>> import xarray as xr
    >>> import xskillscore as xs
    >>> import numpy as np
    >>> # Simulate ensemble forecasts and observations
    >>> fct = xr.DataArray(np.random.rand(10, 5), dims=["time", "member"])
    >>> obs = xr.DataArray(np.random.rand(10), dims=["time"])
    >>> rh = xs.rank_histogram(obs, fct, dim="time")
    >>> fig, ax = xs.plot_rank_histogram(rh)
    >>> plt.show()


    """

    # ----------------------------------------
    # If Dataset → assume exactly one variable
    # ----------------------------------------
    if isinstance(rank_hist, xr.Dataset):
        if len(rank_hist.data_vars) != 1:
            raise ValueError("Dataset must contain exactly one variable.")
        rank_hist = next(iter(rank_hist.data_vars.values()))

    if "rank" not in rank_hist.dims:
        raise ValueError("Input must contain a 'rank' dimension.")

    # --------------------------------------------------
    # Collapse all non-rank dimensions → get 1D histogram
    # --------------------------------------------------
    reduce_dims = [d for d in rank_hist.dims if d != "rank"]
    hist_1d = rank_hist.sum(dim=reduce_dims)

    counts = hist_1d.values.astype(float)

    # Normalize to frequencies if requested
    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total

    ranks = hist_1d["rank"].values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(ranks, counts, width=0.8, edgecolor="black")

    ax.set_xlabel("Rank of Observation Among Ensemble Members")
    ax.set_ylabel("Frequency" if normalize else "Count")

    if title is None:
        title = "Rank Histogram (Talagrand Diagram)"
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(ranks)

    return fig, ax


def plot_rank_histogram_map_cartopy(
    observations: xr.DataArray,
    forecasts: xr.DataArray,
    member_dim: str = "member",
    dim: str = "time",
    rank: int = 1,
    normalize: bool = True,
    use_bias: bool = False,
    projection=ccrs.Robinson(),
    figsize=(14, 6),
):
    """
    Plot a global spatial map of rank-histogram frequency or bias using Cartopy.

    This function visualizes the spatial distribution of a selected rank in the rank histogram,
    either as raw frequency or as bias (deviation from expected frequency), for ensemble forecasts.

    Parameters
    ----------
    observations : xr.DataArray
        Verification observations. Must have dimensions matching `dim` and spatial
        dimensions (e.g., 'lat', 'lon').
    forecasts : xr.DataArray
        Ensemble forecasts. Must have dimensions matching `dim`, `member_dim`, and
        spatial dimensions.
    member_dim : str, default="member"
        Name of the ensemble member dimension in `forecasts`.
    dim : str, default="time"
        Name of the dimension over which to compute the rank histogram (typically time).
    rank : int, default=1
        The rank to plot (1-based index). For example, rank=1 shows where the observation
        falls below all ensemble members.
    normalize : bool, default=True
        If True, show frequency (proportion of times the selected rank occurs).
        If False, show raw counts.
    use_bias : bool, default=False
        If True, plot the bias (frequency minus expected frequency under a uniform distribution).
        If False, plot the raw frequency.
    projection : cartopy.crs.Projection, default=ccrs.Robinson()
        Cartopy projection to use for the map.
    figsize : tuple, default=(14, 6)
        Figure size in inches.

    Returns
    -------
    field : xr.DataArray
        The spatial field of frequency or bias for the selected rank.
        Has the same spatial dimensions as the input.

    Notes
    -----
    - Raw Frequency mode: (`use_bias=False`): “Where does rank 1 happen often?”
        - Plots the frequency (proportion)of the selected rank at each grid point.
        - Useful for identifying where a particular rank (e.g., rank 1: observation below all members) occurs often.
        - The colormap is "viridis"
        - The colorbar is not centered at zero.
    - Bias mode: (`use_bias=True`): “Where is model skewed low/high?”
        - Plots the deviation of the observed frequency from the expected frequency under a uniform distribution.
        - Useful for identifying spatial biases in the ensemble.
        - The colormap is "RdBu_r"
        - The colorbar is centered at zero.
    - The function currently only supports plotting a single rank at a time.


    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from xskillscore.core.viz import plot_rank_histogram_map_cartopy
    >>> # Example data: 10 time steps, 5 members, 20x40 grid
    >>> obs = xr.DataArray(np.random.rand(10, 20, 40), dims=("time", "lat", "lon"))
    >>> fc = xr.DataArray(np.random.rand(10, 5, 20, 40), dims=("time", "member", "lat", "lon"))
    >>> plot_rank_histogram_map_cartopy(obs, fc, member_dim="member", dim="time", rank=1)

    """

    # Align coordinates
    observations, forecasts = xr.align(observations, forecasts, join="inner")

    # Compute histogram
    rh = rank_histogram(
        observations=observations,
        forecasts=forecasts,
        dim=dim,
        member_dim=member_dim,
        keep_attrs=False,
    )
    n_ranks = rh.sizes["rank"]

    freq = rh.isel(rank=rank - 1) / observations.sizes[dim]

    # Choose colormap
    if use_bias:
        expected = 1 / n_ranks
        field = freq - expected
        vmax = np.abs(field).max()
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        field = freq
        vmin = 0.0
        vmax = float(field.max())
        cmap = "viridis"

    # TODO: add these capabilities...
    # Spread/Uncertainty	“Where is ensemble flat?”	cividis	Good for accessibility
    # Probabilistic skill	“Which areas are improving?”	PuOr_r	Strong contrast for ± skill

    # Meshgrid
    lat = field["lat"].values
    lon = field["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Plot
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.coastlines(linewidth=0.7)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    pcm = ax.pcolormesh(
        lon2d,
        lat2d,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    plt.colorbar(
        pcm,
        orientation="horizontal",
        pad=0.04,
        fraction=0.05,
        label="Bias (freq - expected)" if use_bias else "Frequency",
    )

    plt.title(f"Rank Histogram Spatial Map (Rank={rank}, {'bias' if use_bias else 'frequency'})")
    plt.show()

    return field
