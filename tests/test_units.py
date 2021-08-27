import pytest
import numpy as np
import xarray as xr
from cmip6_omz.units import (
    convert_o2_ml_l,
    convert_mol_m3_mymol_kg,
    convert_mg_l_to_mymol_kg
)

###Tests for cmip6-omz.units###

@pytest.fixture
def dummy_o2_ds():
    # create dummy dataset
    ds = xr.DataArray(np.random.rand(4)).to_dataset(name="dummy")
    ds.attrs = {"units": "dummy unit"}
    return ds

def test_convert_o2_ml_l(dummy_o2_ds):
    ds = convert_o2_ml_l(dummy_o2_ds)
    assert ds.attrs["units"] == r"$mol/kg$"
    assert (ds.dummy == dummy_o2_ds.dummy * 43.570 / 1e6).all()

def test_convert_mol_m3_mymol_kg(dummy_o2_ds):
    rho_0 = 1025.
    ds = convert_mol_m3_mymol_kg(dummy_o2_ds)
    assert ds.attrs["units"] == r"$\mu mol/kg$"
    assert (ds.dummy == dummy_o2_ds.dummy / rho_0 * 1e6).all()

def test_convert_mg_l_to_mymol_kg(dummy_o2_ds):
    rho_0 = 1025.
    ds = convert_mg_l_to_mymol_kg(dummy_o2_ds)
    assert ds.attrs["units"] == r"$\mu mol/kg$"
    assert (ds.dummy == dummy_o2_ds.dummy * 1/32000 * rho_0/1000 * 1e6).all()

