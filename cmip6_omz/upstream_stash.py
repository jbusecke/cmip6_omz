import numpy as np
import xarray as xr
from xgcm import Grid
import cf_xarray
import warnings

def sigma_spacing():
    """Return a fine global density (sigma_0) spacing"""
    dsigma = 0.05
    # fine values may not be suitable for time resolved version
    # will result in very large files
    return np.hstack(
        [
            np.array([19.0 - (dsigma / 2)]),
            np.arange(24.5 - (dsigma / 2), 28.0 + (dsigma / 2), dsigma),
            np.array([30.0 + (dsigma / 2)]),
        ]
    )


def xgcm_transform_wrapper(ds, extensive_vars=[], target=None, target_data=None):
    """Return dataset binned in [target_data] space"""

    warnings.warn(
        "xgcm_transform_wrapper() function will be migrated to xgcm. Check xgcm soon for maintained version",
        FutureWarning
    )

    if target_data is None:
        target_data = ds.sigma_0  # only useful for this specific

    target_name = target_data.name

    # remap into density spacing
    if target is None:
        target_raw = sigma_spacing()
    else:
        target_raw = target
    target_bounds = xr.DataArray(
        target_raw, dims=[target_name], coords={target_name: target_raw}
    )

    # Find center of target bins using bounds
    target_centers_raw = (target_bounds.data[1:] + target_bounds.data[0:-1]) / 2
    target_centers = xr.DataArray(
        target_centers_raw, dims=[target_name], coords={target_name: target_centers_raw}
    )

    # for xgcm: we need a way to ignore variables that do not have a dimension on the axis.
    # - need a keepattr option

    # !!! This should be handled internally, because if they are not named the same, we run into problems (huge broadcasted arrays)
    grid = Grid(
        ds, periodic=False, coords={"Z": {"center": "lev", "outer": "lev_outer"}}
    )
    transformed_ds = xr.Dataset()
    # special treatment for sigma_0 (the name conflicts with the new coordinate)
    for var in [
        v for v in ds.data_vars if v != target_name and "lev" in list(ds[v].dims)  #
    ]:  # Drop all variables that do not have a z axis (this should be done in xgcm automatically)
        if var in extensive_vars:
            transformed_ds[var] = grid.transform(
                ds[var],
                "Z",
                target_bounds,
                target_data=target_data,
                method="conservative",
            )
        else:
            transformed_ds[var] = grid.transform(
                ds[var], "Z", target_centers, target_data=target_data
            )
    # Needs to be generalized... add all coordinates
    transformed_ds = transformed_ds.assign_coords(
        {
            co: ds[co]
            for co in [
                c
                for c in ds.coords
                if "lev" not in list(ds[c].dims)
                and "lev_outer"
                not in list(
                    ds[c].dims
                )  # this could be easily excluded with the axis properties
            ]
        }
    )
    ##### the depth of the isopycnal centers is an easy interpolation
    # this doesnt result in a dask array (ask about this on github?)
    #         _, lev_broadcasted = xr.broadcast(ds['sigma_0'],ds.coords['lev'])
    lev_broadcasted = xr.ones_like(target_data) * ds.lev

    #### Transform the main dataset
    transformed_ds = transformed_ds.assign_coords(
        lev=grid.transform(
            lev_broadcasted, "Z", target_centers, target_data=target_data
        )
    )

    lev_vertices = grid.transform(
        lev_broadcasted,
        "Z",
        target_bounds,
        target_data=target_data,
        method="linear",
    )
    # I would like to refactor this with cf-xarray (https://github.com/xarray-contrib/cf-xarray/issues/163)
    lev_bounds = xr.concat(
        [
            lev_vertices.isel({target_name: slice(0, -1)}).assign_coords(
                {target_name: target_centers}
            ),
            lev_vertices.isel({target_name: slice(1, None)}).assign_coords(
                {target_name: target_centers}
            ),
        ],
        dim="bnds",
    )

    ###### how about the thickness? I think we just need to conservatively interpolate the dz values...
    dz_t_broadcasted = xr.ones_like(target_data) * ds.dz_t
    transformed_ds = transformed_ds.assign_coords(
        dz_t=grid.transform(
            dz_t_broadcasted,
            "Z",
            target_bounds,
            target_data=target_data,
            method="conservative",
        ),
        lev_bounds=lev_bounds,
    )

    # paste attrs
    transformed_ds.attrs.update(ds.attrs)

    # This works well. Need to generalize though
    transformed_ds = transformed_ds.assign_coords(
        {
            target_name
            + "_bounds": cf_xarray.vertices_to_bounds(
                target_bounds, out_dims=["bnds", target_name]
            )
        }
    )

    # add the sigma bounds in cf compliant form
    if target_name + "_bounds" not in transformed_ds.coords:
        raise RuntimeError("Implement this thing before saving any more output!")
    return transformed_ds
