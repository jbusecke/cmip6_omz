import numpy as np
import xarray as xr
import pytest
from cmip6_omz.boundaries import omz_boundaries


@pytest.mark.parametrize("vertical_dim", ["lev", "something"])
def test_omz_boundaries(vertical_dim):
    # Create test 1D dataset
    z_raw = np.arange(10) * 3
    o2_raw = np.array([14.0, 13, 12, 10, 0, 3, 0, 2, 2, 3])
    th = 2.7

    o2 = xr.DataArray(o2_raw, dims=[vertical_dim], coords={vertical_dim: z_raw})

    omz_bounds = omz_boundaries(o2[vertical_dim], o2, th, vertical_dim=vertical_dim)

    expected = xr.Dataset(
        {
            f"o2_min_{vertical_dim}": 12.0,
            "o2_min_value": 0.0,
            "upper_boundary": 11.19,
            "lower_boundary": 14.7,
        }
    )

    xr.testing.assert_allclose(expected, omz_bounds)
