import numpy as np
import xarray as xr
from xgcm import Grid
import cf_xarray
import warnings
import collections
import os
import fnmatch
import shutil
import zarr
from rechunker.api import rechunk
import pathlib



from cmip6_preprocessing.drift_removal import remove_trend

# def sigma_spacing():
#     """Return a fine global density (sigma_0) spacing"""
#     dsigma = 0.05
#     # fine values may not be suitable for time resolved version
#     # will result in very large files
#     return np.hstack(
#         [
#             np.array([19.0 - (dsigma / 2)]),
#             np.arange(24.5 - (dsigma / 2), 28.0 + (dsigma / 2), dsigma),
#             np.array([30.0 + (dsigma / 2)]),
#         ]
#     )

def transform_wrapper(
    ds_in,
    intensive_vars=[
        "thetao",
        "o2",
        "so",
        "agessc",
    ],
    sigma_bins=None
):
    """This one stays here? Might be too specific for xgcm for now."""
    
#     sigma_bins = fine_sigma_bins
    #sigma_bins = np.array([0, 24.5, 26.5, 27.65, 100])
    #sigma_bins = np.array([0, 23.0, 24.5, 25.5, 26.5, 26.65, 26.7, 27.4, 27.65, 27.8, 100])
    # define variables to be averaged (intensive quantities)
    intensive_vars = [
        "thetao",
        "o2",
        "so",
        "agessc",
    ]  # add 'uo', 'agessc' etc?

    intensive_vars = [v for v in intensive_vars if v in ds_in.data_vars]

    for iv in intensive_vars:
        dz = (xr.ones_like(ds_in[iv]) * ds_in.dz_t).where(~np.isnan(ds_in[iv]))
        ds_in[iv] = ds_in[iv] * dz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_out = xgcm_transform_wrapper(
            ds_in,
            extensive_vars=["omz_thickness"] + intensive_vars,
            target=sigma_bins,
        )

    # reconvert the same variables
    dz = ds_out.dz_t
    for iv in intensive_vars:
        ds_out[iv] = ds_out[iv] / dz
    return ds_out



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

# modified from this: https://stackoverflow.com/a/37704379
def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def load_data_to_nested_dict(path, match=None, sep="-", **kwargs):
    flist = os.listdir(os.path.join(path))

    if not match is None:
        flist_match = []
        if isinstance(match, str):
            match = [match]
        for m in match:
            flist_match = flist_match + [f for f in flist if fnmatch.fnmatch(f, m)]
        flist = flist_match

    # initialize dict
    out_dict = {}
    for f in flist:  # add a fastprogress bar
        print("Loading %s" % f)
        f_clean = os.path.splitext(f)[0]
        k_list = f_clean.split(sep)
        f_path = os.path.join(path, f)
        nested_set(out_dict, k_list, xr.open_zarr(f_path, **kwargs))
    return out_dict


def flatten_dict(d, parent_key="", sep="-"):
    """flatten a dict by concatenating nested keys with seperator `sep`"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def rechunker_wrapper(source_store, target_store, temp_store, chunks=None, mem="2GiB", consolidated=False, verbose=False):
    # 4GB is based on a node on tigercpu (40 cores/192GB RAM)
    # but wait, should this be per worker? Let me adjust that higher...

    # convert str to paths
    def maybe_convert_to_path(p):
        if isinstance(p, str):
            return pathlib.Path(p)
        else:
            return p

    source_store = maybe_convert_to_path(source_store)
    target_store = maybe_convert_to_path(target_store)
    temp_store = maybe_convert_to_path(temp_store)

    # erase target and temp stores
    if temp_store.exists():
        shutil.rmtree(temp_store)

    if target_store.exists():
        shutil.rmtree(target_store)


    if isinstance(source_store, xr.Dataset):
        g = source_store  # trying to work directly with a dataset
        ds_chunk = g
    else:
        g = zarr.group(str(source_store))
        # get the correct shape from loading the store as xr.dataset and parse the chunks
        ds_chunk = xr.open_zarr(str(source_store))
        
    # preprocess chunks
    if chunks is None:
        chunks = standard_chunks()
    # this should be able to parse -1 as full length chunks (maybe implement that upstream)
#     for di, ch in chunks.items():
#         if ch == -1:
#             chunks[di] = len(source_store[di])

    # convert all paths to strings
    source_store = str(source_store)
    target_store = str(target_store)
    temp_store = str(temp_store)

    group_chunks = {}
    # old version in tuples which is needed for dataset input (https://github.com/pangeo-data/rechunker/issues/59)
    #     for var in ds_chunk.variables:
    #         # pick appropriate chunks from above, and default to full length chunks for dimensions that are not in `chunks` above.
    #         group_chunks[var] = tuple([chunks[di] if di in chunks.keys() else len(ds_chunk[di]) for di in ds_chunk[var].dims])

    # this is the better way to do it...(I should integrate this in the rechunker repo)
    #     for var in ds_chunk.variables:
    #         # pick appropriate chunks from above, and default to full length chunks for dimensions that are not in `chunks` above.
    #         group_chunks[var] = {}
    #         for di in ds_chunk[var].dims:
    #             if di in chunks.keys():
    #                 group_chunks[var][di] = chunks[di]
    #             else:
    #                 group_chunks[var][di] = len(ds_chunk[di])

    # newer tuple version that also takes into account when specified chunks are larger than the array
    for var in ds_chunk.variables:
        # pick appropriate chunks from above, and default to full length chunks for dimensions that are not in `chunks` above.
        group_chunks[var] = []
        for di in ds_chunk[var].dims:
            if di in chunks.keys():
                if chunks[di] > len(ds_chunk[di]):
                    group_chunks[var].append(len(ds_chunk[di]))
                else:
                    group_chunks[var].append(chunks[di])

            else:
                group_chunks[var].append(len(ds_chunk[di]))

        group_chunks[var] = tuple(group_chunks[var])
    if verbose:
        print(f"Rechunking to: {group_chunks}")
    rechunked = rechunk(g, group_chunks, mem, target_store, temp_store=temp_store)
    rechunked.execute()
    if consolidated:
        if verbose:
            print('consolidating metadata')
        zarr.convenience.consolidate_metadata(target_store)
    if verbose:
        print('removing temp store')
    shutil.rmtree(temp_store)
    if verbose:
        print('done')


def rechunk_to_temp(
    source,
    store_target,
    store_temp=None,
    chunks={"time": 6000, "lev": 100, "x":200,"y": 1}, #This cant handle -1 yet...
    consolidated=False,
    overwrite=True,
    mem="1536 MiB"):
    
    
    if not overwrite and store_target.exists():
        print('rechunk_to_temp:rechunked store exists. Not overwriting')
    else:
        for st in [store_temp, store_target]:
            if st.exists():
                shutil.rmtree(st)

    #     # Hmmm this is a bit confusing...
    #     if store_temp is None:
    #         store_temp=ofolder.joinpath("rechunker_temp.zarr")

        variables = list(source.variables)
    #     print(variables)
    #     for var in variables:
    #         if "chunks" in source[var].encoding.keys():
    #             del source[var].encoding["chunks"]
        # TODO: Check if this still hasnt been addressed in xarray?
        rechunker_wrapper(
            source,
            store_target,
            store_temp,
            chunks=chunks,
            mem=mem,
            consolidated=consolidated,
        )
    return xr.open_zarr(store_target, use_cftime=True, consolidated=consolidated)

#==============
# cmip6_pp mods
#==============
import numpy as np
import xarray as xr
from cmip6_preprocessing.postprocessing import exact_attrs, combine_datasets

# # rewrite concat_members
# def concat_members(
#     ds_dict,
#     match_attr_ignore=[],
#     concat_kwargs={},
# ):
#     """Given a dictionary of datasets, this function merges all available ensemble members
#     (given in seperate datasets) into a single dataset for each combination of attributes,
#     like source_id, grid_label, etc. but with concatnated members.
#     CAUTION: If members do not have the same dimensions (e.g. longer run time for some members),
#     this can result in poor dask performance (see: https://github.com/jbusecke/cmip6_preprocessing/issues/58)
#     Parameters
#     ----------
#     ds_dict : dict
#         Dictionary of xarray datasets.
#     concat_kwargs : dict
#         Optional arguments passed to xr.concat.
#     Returns
#     -------
#     dict
#         A new dict of xr.Datasets with all datasets from `ds_dict`, but with concatenated members and adjusted keys.
#     """
#     # TODO: convert str to list maybe
#     match_attr_ignore.extend(['variant_label'])
#     match_attrs = [ma for ma in exact_attrs if ma not in match_attr_ignore]

#     # set defaults
#     concat_kwargs.setdefault(
#         "combine_attrs", "drop_conflicts"
#     )  # if the size differs throw an error. Requires xarray >=0.17.0

#     return combine_datasets(
#         ds_dict,
#         xr.concat,
#         combine_func_args=(
#             ["member_id"]
#         ),  # I dont like this. Its confusing to have two different dimension names
#         combine_func_kwargs=concat_kwargs,
#         match_attrs=match_attrs,
#     )


# def _pick_first_member(ds_list, **kwargs):
#     idx = 0
#     # only pick the ones that are fully concatenated
#     while (
#         str(ds_list[idx].time.to_index()[-1]) < "2090"
#         or str(ds_list[idx].time.to_index()[0]) > "1901"
#     ):
#         # print(ds_list[idx].attrs)
#         idx += 1
#     return ds_list[idx]
def _pick_first_member(ds_list, **kwargs):
    members = [ds.variant_label for ds in ds_list]
    first_member_idx = np.argmin(members)
    return ds_list[first_member_idx]


def pick_first_member(ddict):
    return combine_datasets(
        ddict,
        _pick_first_member,
        match_attrs=["source_id", "grid_label", "table_id"],
    )

def _maybe_str_to_list(a):
    if isinstance(a, list):
        return a
    else:
        return [a]


# custom define function that sorts input by time...
def _concat_sorted_time(ds_list, **kwargs):
    # extract the first date
    start_dates = [str(ds.time.to_index()[0]) for ds in ds_list]
    sorted_idx = np.argsort(start_dates)
    ds_list_sorted = [ds_list[i] for i in sorted_idx]
    return xr.concat(ds_list_sorted, "time", **kwargs)


def concat_experiments(
    ds_dict,
    exclude_attrs=[],
    concat_kwargs={},
):
    """Given a dictionary of datasets, this function merges all available ensemble members
    (given in seperate datasets) into a single dataset for each combination of attributes,
    like source_id, grid_label, etc. but with concatnated members.
    CAUTION: If members do not have the same dimensions (e.g. longer run time for some members),
    this can result in poor dask performance (see: https://github.com/jbusecke/cmip6_preprocessing/issues/58)
    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray datasets.
    exclude_attrs : list
        List of attributes that should be excluded from matching. This is necessary to nest different
        combination wrappers (which might eliminate certain attributes in the process).
    concat_kwargs : dict
        Optional arguments passed to xr.concat.
    Returns
    -------
    dict
        A new dict of xr.Datasets with all datasets from `ds_dict`, but with concatenated members and adjusted keys.
    """
    exclude_attrs = _maybe_str_to_list(exclude_attrs)

    match_attrs = [
        ma for ma in exact_attrs if ma not in ["experiment_id"] + exclude_attrs
    ]

    # set defaults
    concat_kwargs.setdefault(
        "combine_attrs",
        "drop_conflicts",
    )  # if the size differs throw an error. Requires xarray >=0.17.0
    concat_kwargs.setdefault("compat", "override")
    concat_kwargs.setdefault("coords", "minimal")

    return combine_datasets(
        ds_dict,
        _concat_sorted_time,
        combine_func_kwargs=concat_kwargs,
        match_attrs=match_attrs,
    )


import cf_xarray
def construct_static_dz(ds):
    lev_vertices = cf_xarray.bounds_to_vertices(ds.lev_bounds, 'bnds').load()
    dz_t = lev_vertices.diff('lev_vertices')
    ds = ds.assign_coords(thkcello=('lev', dz_t.data))
    return ds
    