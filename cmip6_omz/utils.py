import socket

def cmip6_collection(zarr=False):
    """returns the cmip6 collection/catalog for the current host"""
    hostname = socket.gethostname()
    if "tiger" in hostname:
        if zarr:
            url = "/home/jbusecke/projects/cmip_data_management_princeton/catalogs/tigercpu-zarr-cmip6.json"
        else:
            url = "/home/jbusecke/projects/cmip_data_management_princeton/catalogs/tigercpu-cmip6.json"
    elif "tigress" in hostname:
        if zarr:
            url = "/home/jbusecke/projects/cmip_data_management_princeton/catalogs/tigressdata-zarr-cmip6.json"
        else:
            url = "/home/jbusecke/projects/cmip_data_management_princeton/catalogs/tigressdata-cmip6.json"
    elif "jupyter-" in hostname:
        url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    return url


######## These are temp, and should be replaced eventually #########
from aguadv_omz_busecke_2021.cmip6_stash import (
    load_single_datasets,
    load_trend_dict,
    match_and_detrend,
)
import pathlib
import intake

def parse_member_id(name, ds, verbose=True):
    """Shoule be deleted when the attributes are correctly set."""
    if ds is not None:
        member_id = ds.attrs.get('variant_label')
        # if attrs not available use the name (less reliable in general)
        if member_id is None:
            warnings.warn('Parsing member from name. This is risky')
            member_id = name.split('.')[-2]
        
        if verbose:
            if 'member_id' not in ds.dims:
                print(f"no member id in dims for {name}")
            else:
                print(f"Member Parsing: Found {ds.member_id.data} for {name}")

        ds = ds.assign_coords(member_id=xr.DataArray([member_id], dims=['member_id']))
        return ds

def loading_wrapper(
    detrend=False,
    trendfolder = pathlib.Path('../../data/processed/linear_regression_time_zarr_multimember/'),
    pass_variables=["mlotst"],
    metrics=None,
    nest=False,
    verbose=False,
    fix_member_id=False, 
    **kwargs):
    """Monster Wrapper to unify the following things across all notebooks:
    1. Loading the data
    2. Detrending when applicable
    """
    print('### Loading Raw Data ###')
    col = intake.open_esm_datastore(cmip6_collection(zarr=True))
    ddict = load_single_datasets(col,**kwargs)
    if detrend:
        print('### Detrending Data ###')
        trend_dict = load_trend_dict(trendfolder, ddict, verbose=verbose)
        ddict_detrended = match_and_detrend(
            ddict,
            trend_dict,
            verbose=verbose,
            pass_variables=pass_variables,
        )
        
        ddict_detrended_filtered = {k:v for k,v in ddict_detrended.items() if 'rho' not in v.dims}
        ddict = ddict_detrended_filtered

    if fix_member_id:
        ddict_fixed_members = {k:parse_member_id(k,ds, verbose=verbose) for k, ds in ddict.items()}
        ddict = ddict_fixed_members
    return ddict