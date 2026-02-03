import numpy as np
import pytest
import xarray as xr

from xskillscore.core.deterministic import anomaly_correlation_coefficient


# ======================================================================
# Manual ACC
# ======================================================================
def manual_acc_point(obs_ts, fct_ts, clim_ts=None):
    obs = obs_ts.values
    fct = fct_ts.values

    if clim_ts is not None:
        clim = clim_ts.values
        obs_anom = obs - clim
        fct_anom = fct - clim
    else:
        obs_anom = obs - obs.mean()
        fct_anom = fct - fct.mean()

    num = np.sum(obs_anom * fct_anom)
    den = np.sqrt(np.sum(obs_anom**2) * np.sum(fct_anom**2))

    if den == 0:
        return np.nan
    return num / den


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def coords():
    times = np.arange("2000-01-01", "2000-01-04", dtype="datetime64[D]")
    lats = np.linspace(-30, 30, 4)
    lons = np.linspace(0, 270, 4)
    return times, lats, lons


@pytest.fixture
def doy(coords):
    times, *_ = coords
    return xr.DataArray(times).dt.dayofyear.values


# ======================================================================
# Builder functions for test cases
# ======================================================================


def build_random_case(coords, doy):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.DataArray(
        np.random.randn(4, nT, nLat, nLon),
        dims=["member", "time", "lat", "lon"],
        coords={"member": ["m0", "m1", "m2", "m3"], "time": times, "lat": lats, "lon": lons},
    )

    clim = xr.DataArray(
        np.random.randn(len(np.unique(doy)), nLat, nLon),
        dims=["dayofyear", "lat", "lon"],
        coords={"dayofyear": np.unique(doy), "lat": lats, "lon": lons},
    )

    return obs, fct, clim


def build_static_clim_case(coords):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.DataArray(
        np.random.randn(4, nT, nLat, nLon),
        dims=["member", "time", "lat", "lon"],
        coords={"member": ["m0", "m1", "m2", "m3"], "time": times, "lat": lats, "lon": lons},
    )

    clim = xr.DataArray(
        np.random.randn(nLat, nLon), dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
    )

    return obs, fct, clim


def build_time_clim_case(coords):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.DataArray(
        np.random.randn(4, nT, nLat, nLon),
        dims=["member", "time", "lat", "lon"],
        coords={"member": ["m0", "m1", "m2", "m3"], "time": times, "lat": lats, "lon": lons},
    )

    clim = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    return obs, fct, clim


def build_no_clim_case(coords):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.DataArray(
        np.random.randn(4, nT, nLat, nLon),
        dims=["member", "time", "lat", "lon"],
        coords={"member": ["m0", "m1", "m2", "m3"], "time": times, "lat": lats, "lon": lons},
    )

    return obs, fct, None


def build_perfect_corr_case(coords):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.concat([obs] * 4, dim="member")
    fct = fct.assign_coords(member=["m0", "m1", "m2", "m3"])

    return obs, fct, None


def build_perfect_anticorr_case(coords):
    obs, _, _ = build_no_clim_case(coords)

    fct = xr.concat([-obs] * 4, dim="member")
    fct = fct.assign_coords(member=["m0", "m1", "m2", "m3"])

    return obs, fct, None


def build_constant_case(coords):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.ones((nT, nLat, nLon)),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.concat([obs] * 4, dim="member")
    fct = fct.assign_coords(member=["m0", "m1", "m2", "m3"])

    clim = xr.DataArray(
        np.zeros((nLat, nLon)), dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
    )

    return obs, fct, clim


def build_nan_case(coords, doy):
    obs, fct, clim = build_random_case(coords, doy)

    obs.values[0, 0, 0] = np.nan
    fct.values[:, 1, 1, 1] = np.nan

    return obs, fct, clim


def build_interp_case(coords, doy):
    times, lats, lons = coords
    nT, nLat, nLon = len(times), len(lats), len(lons)

    obs = xr.DataArray(
        np.random.randn(nT, nLat, nLon),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )

    fct = xr.DataArray(
        np.random.randn(4, nT, nLat, nLon),
        dims=["member", "time", "lat", "lon"],
        coords={"member": ["m0", "m1", "m2", "m3"], "time": times, "lat": lats, "lon": lons},
    )

    # 2x2 grid climatology → tests interpolation via align
    lats2 = np.linspace(-30, 30, 2)
    lons2 = np.linspace(0, 270, 2)

    clim_small = xr.DataArray(
        np.random.randn(len(np.unique(doy)), 2, 2),
        dims=["dayofyear", "lat", "lon"],
        coords={"dayofyear": np.unique(doy), "lat": lats2, "lon": lons2},
    )

    return obs, fct, clim_small


# ======================================================================
# PARAMETRIZED TEST SUITE
# ======================================================================
@pytest.mark.parametrize(
    "builder",
    [
        build_random_case,
        build_static_clim_case,
        build_time_clim_case,
        build_no_clim_case,
        build_perfect_corr_case,
        build_perfect_anticorr_case,
        build_constant_case,
        build_nan_case,
        build_interp_case,
    ],
)
def test_acc_all_configs(builder, coords, doy):
    """Validate ACC correctness across many climatology/forecast/obs configs."""

    # Build dataset
    if builder.__name__ in ["build_random_case", "build_nan_case", "build_interp_case"]:
        obs, fct, clim = builder(coords, doy)
    else:
        obs, fct, clim = builder(coords)

    # Compute ACC library
    acc = anomaly_correlation_coefficient(fct, obs, dim="time", climatology=clim)

    # Manual comparison
    times, lats, lons = coords
    doy_vals = xr.DataArray(times).dt.dayofyear.values

    tol = 1e-6
    for m in range(fct.sizes["member"]):
        for i in range(len(lats)):
            for j in range(len(lons)):
                obs_ts = obs.isel(lat=i, lon=j)
                fct_ts = fct.isel(member=m, lat=i, lon=j)

                if clim is None:
                    clim_ts = None
                else:
                    if "dayofyear" in clim.dims:
                        clim_ts = clim.sel(dayofyear=doy_vals).isel(lat=i, lon=j)
                    elif "time" in clim.dims:
                        clim_ts = clim.isel(lat=i, lon=j)
                    else:
                        clim_ts = clim.isel(lat=i, lon=j).broadcast_like(obs_ts)

                manual = manual_acc_point(obs_ts, fct_ts, clim_ts)
                lib = float(acc.isel(member=m, lat=i, lon=j).values)

                if np.isnan(manual) and np.isnan(lib):
                    continue

                assert np.isclose(manual, lib, atol=tol, rtol=tol), (
                    f"Mismatch in {builder.__name__}(member={m}, lat={i}, lon={j}): manual={manual}, lib={lib}"
                )
