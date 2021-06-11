import pytest
import numpy as np
import xarray as xr
from xarrayutils.utils import dll_dist

from cmip6_omz.upstream_stash import(
    reconstruct_areacello
)

def test_reconstruct_areacello():

    #test case with no bounds 

    lon_ = np.arange(0,360)
    lat_ = np.arange(-89,90)
    lon = xr.DataArray(lon_, dims=['x'],coords={'x': np.arange(len(lon_))}, name='lon')
    lat = xr.DataArray(lat_, dims=['y'],coords={'y': np.arange(len(lat_))}, name='lat')

    ds = xr.Dataset().assign_coords({"lon": lon, "lat": lat})
    area1 = reconstruct_areacello(ds)
    
    #spacing is 1 degree everywhere for test case
    dlon = xr.DataArray(np.ones_like(ds.lon), coords=ds.lon.coords)
    dlat = xr.DataArray(np.ones_like(ds.lat), coords=ds.lat.coords)
    dx, dy = dll_dist(dlon, dlat, ds.lon, ds.lat) #test this function upstream
    area2 = dx * dy

    xr.testing.assert_allclose(area1, area2.squeeze())

    
    
