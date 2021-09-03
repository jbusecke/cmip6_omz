from numba import float64, guvectorize
import numpy as np
import xarray as xr


@guvectorize(
    [
        (float64[:], float64[:], float64, float64[:], float64[:]),
    ],
    "(n),(n),(), (m)->(m)",
    nopython=True,
)
def _find_boundaries(z, o2, threshold, output_size, output):
    # initialize output
    output[:] = np.nan
    # find local minimum
    min_idx = np.argmin(o2)

    output[0] = z[
        min_idx
    ]  # I should probably test if this is at the edges of z (top bottom), probably want ot exclude them?
    output[1] = o2[
        min_idx
    ]  # I should probably test if this is at the edges of z (top bottom), probably want ot exclude them?

    forward_idx = backward_idx = min_idx
    # iterate downwards from z_min, find the first value that is larger than the threshold, record the idx and interpolate linearly
    # I can probably make this more elegant...but for now lets just do it this way

    for idx in range(0, min_idx):
        if o2[idx] > threshold:
            backward_idx = idx
            pass

    # same forward
    for idx in range(min_idx, len(o2)):
        if o2[idx] > threshold:
            forward_idx = idx
            pass

    def sort_and_interp(th, o2, z):
        if o2[-1] < o2[0]:
            o2 = o2[::-1]
            z = z[::-1]
        return np.interp(th, o2, z)

    # if the indexes are still equal to `min_idx` it means
    # that in this direction there was no value smaller than threshold
    # in that case the omz extends to the bottom/top and the values should be set to nan
    if backward_idx == min_idx:
        back_z = np.nan

    else:
        backward_o2 = o2[backward_idx : min_idx + 1]
        backward_z = z[backward_idx : min_idx + 1]
        back_z = sort_and_interp(threshold, backward_o2, backward_z)

    if forward_idx == min_idx:
        forw_z = np.nan
    else:
        forward_o2 = o2[min_idx : forward_idx + 1]
        forward_z = z[min_idx : forward_idx + 1]
        forw_z = sort_and_interp(threshold, forward_o2, forward_z)

    output[2] = back_z
    output[3] = forw_z


def _find_omz_boundaries(z, o2, threshold):
    # this is a bit of a hack, but for now the only way I found to allocate the new array along a new dimension with a fixed length
    output_dummy = np.arange(4)
    return _find_boundaries(z, o2, threshold, output_dummy)


def omz_boundaries(z, o2, o2_threshold, vertical_dim="lev", fill_val=1e32, **kwargs):
    """Analyze the vertical boundaries and o2 minimum.

    Parameters
    ----------
    z : xr.DataArray
        vertical coordinate to find the boundary position on.
        Usually depth, but could also be another tracer (like density)
    o2 : xr.DataArray
        oxygen data input
    o2_threshold : float
        threshold defining the OMZ (in units of `o2`)
    vertical_dim : str, optional
        vertical xarray dimension in `z` and `o2`, by default "lev"
    fill_val : float, optional
        values to fill nans with, needs to be higher than `o2` values everywhere, by default 1e32

    Returns
    -------
    xr.Dataset
        Dataset containing the o2 values and `z` values of the 'omz-core' (minimum `o2` in the vertical),
        and the upper and lower boundary, which is lineraly interpolated between the omz-core and the first
        value > `o2_threshold` along the vertical dimension.
    """
    out = xr.apply_ufunc(
        _find_omz_boundaries,
        z,
        o2.fillna(fill_val),
        o2_threshold,
        kwargs=kwargs,
        input_core_dims=[[vertical_dim], [vertical_dim], []],
        output_core_dims=[["new"]],
        dask="parallelized",
        output_dtypes=[o2.dtype],
        dask_gufunc_kwargs=dict(output_sizes={"new": 4}),
    )
    out = xr.Dataset(
        {
            f"o2_min_{z.name}": out.isel(new=0),
            "o2_min_value": out.isel(new=1),
            "upper_boundary": out.isel(new=2),
            "lower_boundary": out.isel(new=3),
        }
    )
    return out
