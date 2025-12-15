import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xskillscore.core.probabilistic import rank_histogram


def plot_rank_histogram(rank_hist, normalize=True, title=None):
    """
    Plot a Rank Histogram (Talagrand Diagram) from the direct output of
    xskillscore.rank_histogram().

    Parameters
    ----------
    rank_hist : xr.DataArray or xr.Dataset
        Output of xskillscore.rank_histogram(...).
        Must include a 'rank' dimension.

    normalize : bool, default=True
        If True, converts counts to frequencies (sum to 1).

    title : str or None
        Optional custom plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
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
    Plot a global spatial map of rank-histogram frequency or bias.

    Raw Frequency	“Where does rank 1 happen often?”	viridis	Uniform, no zero center needed
    Bias Map	“Where is model skewed low/high?”	RdBu_r	Symmetric around zero
    Spread/Uncertainty	“Where is ensemble flat?”	cividis	Good for accessibility
    Probabilistic skill	“Which areas are improving?”	PuOr_r	Strong contrast for ± skill
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

    # Meshgrid
    lat = field["lat"].values
    lon = field["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Plot
    fig = plt.figure(figsize=figsize)
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
