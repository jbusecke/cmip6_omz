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


@pytest.mark.parametrize("vertical_dim", ["lev", "something"])
def test_omz_boundaries_mulitple(vertical_dim):
    # This is not an efficient way to do this...
    z_raw = np.arange(10) * 3
    o2_raw = np.array([14.0, 13, 12, 10, 0, 3, 0, 2, 2, 3])
    th = np.array([2.7, 5.0, 16])
    th = xr.DataArray(th, dims="o2_threshold", coords={"o2_threshold": th})

    o2 = xr.DataArray(
        o2_raw, dims=[vertical_dim], coords={vertical_dim: z_raw}
    ).expand_dims({"x": 3})

    omz_bounds = omz_boundaries(o2[vertical_dim], o2, th, vertical_dim=vertical_dim)

    expected_coords = {"o2_threshold": th}
    expected = xr.Dataset(
        {
            f"o2_min_{vertical_dim}": xr.DataArray(
                [12.0, 12.0, 12.0], dims=["o2_threshold"]
            ).expand_dims({"x": 3}),
            "o2_min_value": xr.DataArray(
                [0.0, 0.0, 0.0], dims=["o2_threshold"]
            ).expand_dims(
                {"x": 3}
            ),  # these should not be duplicated
            "upper_boundary": xr.DataArray(
                [11.19, 10.5, np.nan],
                dims=["o2_threshold"],
                coords=expected_coords,
            ).expand_dims({"x": 3}),
            "lower_boundary": xr.DataArray(
                [14.7, np.nan, np.nan],
                dims=["o2_threshold"],
                coords=expected_coords,
            ).expand_dims({"x": 3}),
        }
    )

    xr.testing.assert_allclose(expected, omz_bounds)


# def test_omz_boundaries_mulitple(vertical_dim):
#     # THIS is how I would want it to work
#     z_raw = np.arange(10) * 3
#     o2_raw = np.array([14.0, 13, 12, 10, 0, 3, 0, 2, 2, 3])
#     th = np.array([1.2, 2.7, 5.8])
#     th = xr.DataArray(th, dims="o2_threshold", coords={"o2_threshold": th})

#     o2 = xr.DataArray(
#         o2_raw, dims=[vertical_dim], coords={vertical_dim: z_raw}
#     ).expand_dims({"x": 3})

#     omz_bounds = omz_boundaries(o2[vertical_dim], o2, th, vertical_dim=vertical_dim)

#     expected_coords = {"o2_threshold": th}
#     expected = xr.Dataset(
#         {
#             f"o2_min_{vertical_dim}": xr.DataArray(12.0).expand_dims({"x": 3}),
#             "o2_min_value": xr.DataArray(0.0).expand_dims({"x": 3}),
#             "upper_boundary": xr.DataArray(
#                 [11.19, np.nan, np.nan], dims=["o2_threshold"], coords=expected_coords
#             ).expand_dims({"x": 3}),
#             "lower_boundary": xr.DataArray(
#                 [14.7, np.nan, np.nan], dims=["o2_threshold"], coords=expected_coords
#             ).expand_dims({"x": 3}),
#         }
#     )

#     xr.testing.assert_allclose(expected, omz_bounds)
