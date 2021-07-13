import numpy as np
import xarray as xr
from xgcm import Grid
import cf_xarray
import warnings

from cmip6_preprocessing.drift_removal import remove_trend

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


####function to cmip6_pp#########


#Helper functions for match_and_detrend()
#but may also be useful elsewhere
def construct_cfdate(data, units, calendar):
    date = xr.DataArray(data, attrs={"units": units, "calendar": calendar})
    return xr.decode_cf(date.to_dataset(name="time"), use_cftime=True).time

def _get_calendar(time):
    ### This assumes that the time is decoded already
    return time.encoding.get("calendar", "standard")


def match_and_detrend(data_dict, trend_dict, pass_variables=[], verbose=False):
    """
    Takes CMIP6 datasets dict and a dictionary of trend datasets, and returns the detrended datasets.
    Datasets with variable in `pass_variables` are passed through unaltered.
    """
    sep='_'
    
    compare_attrs = [ #should version_id also be here?
        "source_id",
        "table_id",
        "grid_label",
        "variant_label",
        "variable_id",
    ]


    data_dict_detrended = {}
    for name, ds in data_dict.items():
        ds = ds.copy()
        if ds.attrs["variable_id"] in pass_variables:
            if verbose:
                print(f"Manually Ignored Variable: Passing {name} unaltered")
            data_dict_detrended[name] = ds
        elif ds.attrs["experiment_id"] == "piControl": 
            if verbose:
                print(f"Control run: Passing {name} unaltered")
            data_dict_detrended[name] = ds
        else:
            match_elements = [ds.attrs[i] for i in compare_attrs]
            match_targets = list(trend_dict.keys()) #use attr once trend files preserve them
            match_trend_names = [
                m for m in match_targets if all([me+sep in m for me in match_elements])
            ]
            if len(match_trend_names) == 1:
                trend_ds = trend_dict[match_trend_names[0]]

                ref_date = construct_cfdate(
                    0, "hours since 1850-1-15", _get_calendar(ds.time)
                ).data.tolist()
                
                #should this be handled elsewhere?
                if "slope" in trend_ds.variables:
                    trend_ds = trend_ds.rename({"slope":ds.attrs["variable_id"]})
                
                da_detrended = remove_trend(
                    ds, 
                    trend_ds,
                    ds.attrs["variable_id"],
                    ref_date=ref_date
                )
                ds[ds.attrs["variable_id"]] = da_detrended.reset_coords(drop=True)
                data_dict_detrended[name] = ds
                
                ds.attrs["detrended_with_file"] = 'Done'
                
            elif len(match_trend_names) > 1:
                raise ValueError(f"Found multiple matches for {match_elements}: {match_trend_names}. Check input")
            else:
                warnings.warn(f"No match found for {match_elements}.")
    return data_dict_detrended

##############################Misc#############

#These are fixes so that the trend data works with cmip6_pp match_and_remove_trend
#these issues should be addressed in the next iteration of trend file production
def fix_trend_metadata(trend_dict):
    for name, ds in trend_dict.items():
        #restore attributes to trend datasets using file names
        #assumes consistent naming scheme for file names
        fn = (ds.attrs['filepath']).rsplit("/")[-1]
        fn_parse = fn.split('_')
        ds.attrs['source_id'] = fn_parse[2]
        ds.attrs['grid_label'] = fn_parse[5]
        ds.attrs['experiment_id'] = fn_parse[3]
        ds.attrs['table_id'] = fn_parse[4]
        ds.attrs['variant_label'] = fn_parse[7]
        ds.attrs['variable_id'] = fn_parse[8]
        
        #rename 'slope' variable to variable_id
        if "slope" in ds.variables:
            ds = ds.rename({"slope":ds.attrs["variable_id"]})
        
        #error was triggered in line 350 of cmip6_preprocessing.drift_removal
        ##this is a temporary workaround, and the one part of this function that might
        ##require an upstream fix (though it might just be an environment issue)
        #ds = ds.drop('trend_time_range')
        
        trend_dict[name] = ds
        
    return trend_dict
