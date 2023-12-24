import numpy as np
import pytest
import xarray as xr

from cmip6_omz.omz_tools import (
    full_volume,
    mask_basin,
    omz_full_volume,
    omz_thickness,
    sample_select,
    volume_consistency_checks,
)


###Testing suite for cmip6_omz.omz_tools###


# dummy xr dataset in density space
# with dz, dy, dx equal to 1. everywhere
@pytest.fixture
def dummy_ds():
    nx = 50
    ny = 50
    n_sig = 10
    nt = 50
    x = np.arange(nx)
    y = np.arange(ny)
    sigma_0 = np.linspace(20, 30, n_sig)
    t = np.arange(nt)
    np.random.seed(0)
    ds = xr.DataArray(
        (300.0 * 1e6 / 1025.0) * np.random.random((nt, n_sig, ny, nx)),
        dims=["time", "sigma_0", "y", "x"],
        coords={"x": x, "y": y, "sigma_0": sigma_0, "time": t},
    ).to_dataset(name="o2")

    dx = xr.ones_like(ds.o2.isel(time=0, sigma_0=0))
    ds["dx_t"] = dx
    ds["dy_t"] = dx
    ds["dz_t"] = xr.ones_like(ds.o2.isel(time=0, x=0, y=0))
    return ds


# dummy xr dataset with multiple members
@pytest.fixture
def dummy_ds_2mem():
    nx = 10
    ny = 10
    nz = 10
    nt = 50
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    t = np.arange(nt)
    np.random.seed(0)
    ds1 = xr.DataArray(
        (300.0 * 1e6 / 1025.0) * np.random.random((nt, nz, ny, nx)),
        dims=["time", "z", "y", "x"],
        coords={"x": x, "y": y, "z": z, "time": t},
    ).to_dataset(name="o2")

    np.random.seed(1)
    ds2 = xr.DataArray(
        (300.0 * 1e6 / 1025.0) * np.random.random((nt, nz, ny, nx)),
        dims=["time", "z", "y", "x"],
        coords={"x": x, "y": y, "z": z, "time": t},
    ).to_dataset(name="o2")

    ds = xr.concat([ds1, ds2], dim="member_id")

    return ds


###Test functions###


def test_sample_select(dummy_ds_2mem):
    # function should isolate first member
    # and slice time through (12,48)

    ds = sample_select(dummy_ds_2mem)
    assert "member_id" not in ds.dims
    assert len(ds.time) == 36


@pytest.mark.parametrize("o2_bins", [5, 10, 20, 40, 60, 80, 100, 120, 140])
def test_omz_thickness(dummy_ds, o2_bins):
    # check using simple case of dz = 1 everywhere

    ds = dummy_ds.copy()
    ds_test = dummy_ds.copy()

    ds["omz_thickness"] = omz_thickness(ds, o2_bins=np.array([o2_bins]))

    # redo calculation manually and compare
    o2 = dummy_ds.o2.copy() / 1025 * 1e6
    dz = xr.ones_like(o2)
    thickness_test = [
        dz.where(o2 <= o2_bins, 0).assign_coords(o2_bin=o2_bins).astype(o2.dtype)
    ]
    ds_test["omz_thickness"] = xr.concat(thickness_test, dim="o2_bin")

    xr.testing.assert_equal(ds_test, ds)


def test_full_volume(dummy_ds):
    # dummy dataset has no areacello, so should use dx,dy
    vol = full_volume(dummy_ds)

    # redo calculation manually and compare
    ds_test = sample_select(dummy_ds)
    area_test = ds_test.dx_t * ds_test.dy_t
    dz_test = xr.ones_like(ds_test.o2) * ds_test.dz_t
    vol_test = dz_test * area_test
    vol_test = vol_test.where(~np.isnan(ds_test.sigma_0))

    xr.testing.assert_equal(vol, vol_test)


def test_omz_full_volume(dummy_ds):
    dummy_ds["omz_thickness"] = omz_thickness(dummy_ds)
    omz_vol = omz_full_volume(dummy_ds)

    # redo calculation manually and compare
    ds_test = sample_select(dummy_ds)
    area_test = ds_test.dx_t * ds_test.dy_t
    omz_vol_test = ds_test.omz_thickness * area_test

    xr.testing.assert_equal(omz_vol, omz_vol_test)


# not sure how to test this without a script
# to transform from z-space to sigma-space
# def test_volume_consistency_checks():


@pytest.mark.parametrize("basins", ["Pacific", "Atlantic", "Indian"])
def test_mask_basin(basins):
    nx = 360
    ny = 180
    n_sig = 10
    lon = np.arange(nx)
    lat = np.arange(-90, 90)
    sigma_0 = np.linspace(20, 30, n_sig)
    np.random.seed(0)
    ds = xr.DataArray(
        (300.0 * 1e6 / 1025.0) * np.random.random((n_sig, ny, nx)),
        dims=["sigma_0", "lat", "lon"],
        coords={"lon": lon, "lat": lat, "sigma_0": sigma_0},
    ).to_dataset(name="o2")

    ds_masked = mask_basin(ds, region=basins)

    # check that masks crop horizontally
    # and don't change the vertical
    assert len(ds_masked.lon) < len(ds.lon)
    assert len(ds_masked.lat) < len(ds.lat)
    assert len(ds_masked.sigma_0) == len(ds.sigma_0)
