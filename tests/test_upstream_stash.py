import pytest
import numpy as np
import xarray as xr
import cf_xarray

from cmip6_omz.upstream_stash import xgcm_transform_wrapper, construct_static_dz
from cmip6_omz.omz_tools import omz_thickness

@pytest.fixture
def dummy_ds():
    #mock up test dataset
    nx = 10
    ny = 10
    nz = 10
    nt = 10

    x = np.arange(nx)
    y = np.arange(ny)
    lev = np.arange(nz)
    t = np.arange(nt)

    da_o2 = xr.DataArray(
        (300.*1e6 / 1025.)*np.random.random((nt, nz, ny, nx)),
        dims=["time", "lev", "y", "x"],
        coords={"x": x, "y": y, "lev":lev, "time": t},
    ).to_dataset(name = "o2")

    da_sigma0 = xr.DataArray(
        (24. + 5.*np.random.random((nt, nz, ny, nx))),
        dims=["time", "lev", "y", "x"],
        coords={"x": x, "y": y, "lev":lev, "time": t},
    ).to_dataset(name = "sigma_0")

    ds = xr.merge([da_o2, da_sigma0])
    ds['dz_t'] = xr.ones_like(ds.o2)
    ds['areacello'] = xr.ones_like(ds.o2)
    return ds


def test_xgcm_transform_wrapper(dummy_ds):

    ds = dummy_ds
    sigma_bins = np.linspace(24, 28, 5)
    ds_sigma = xgcm_transform_wrapper(
        ds, 
        target = sigma_bins,
        target_data = ds.sigma_0
    )

    assert('sigma_0' in ds_sigma.o2.dims)
    assert('lev' not in ds_sigma.o2.dims)

def test_xgcm_transform_wrapper2(dummy_ds):
    #integration test with cmip6_omz.omz_tools.omz_thickness()

    ds = dummy_ds
    ds['omz_thickness'] = omz_thickness(ds)
    sigma_bins = np.linspace(24, 28, 5)

    ds_sigma = xgcm_transform_wrapper(
        ds, 
        extensive_vars = ['omz_thickness'],
        target = sigma_bins,
        target_data = ds.sigma_0
    )

    assert('sigma_0' in ds_sigma.omz_thickness.dims)
    assert('lev' not in ds_sigma.omz_thickness.dims)
    
    #test if omz volume is conserved within 1 percent
    vol_sigma = (ds_sigma.omz_thickness * ds_sigma.areacello).sum(['x','y','sigma_0']).load()
    vol_z = (ds.omz_thickness * ds.areacello).sum(['x', 'y', 'lev']).load()

    xr.testing.assert_allclose(vol_sigma, vol_z, rtol = 1e-02) 

    
def test_construct_static_dz():
    lev_verts = np.array([0, 10, 250, 4000])
    lev_bounds = xr.DataArray([lev_verts[:-1],lev_verts[1:]], dims=['bnds', 'lev'])
    ds = xr.Dataset().assign_coords(lev_bounds=lev_bounds)
    expected = xr.DataArray([10, 240, 3750], dims=['lev'], name='thkcello')
    
    ds_reconstructed = construct_static_dz(ds)
    xr.testing.assert_equal(ds_reconstructed.thkcello.reset_coords(drop=True), expected)
