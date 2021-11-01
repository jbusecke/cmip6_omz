import socket
import intake
import pandas as pd


### temporary fix needed to avoid duplicate versions (see https://github.com/NCAR/intake-esm-datastore/pull/84)
def _pick_latest_version_aggressive(df):
    import itertools

    print(f'Dataframe size before picking latest version: {len(df)}')
    grpby = list(set(df.columns.tolist()) - {'path', 'version', 'dcpp_init_year', 'time_range'}) # I cannot do this generally, because then we throw out actual data
    grouped = df.groupby(grpby)

    def _pick_latest_v(group):
        idx = []
        if group.version.nunique() > 1:
            idx = group.sort_values(by=['version'], ascending=False).index[1:].values.tolist()
        return idx

    print('Getting latest version...\n')
    idx_to_remove = grouped.apply(_pick_latest_v).tolist()
    idx_to_remove = list(itertools.chain(*idx_to_remove))
    df = df.drop(index=idx_to_remove)
    print(f'Dataframe size after picking latest version: {len(df)}')
    print('\nDone....\n')
    return df

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
    elif "jupyter.rc" in hostname:
        # testing if this works on jupyterrc?
        if zarr:
            url = "/projects/GEOCLIM/LRGROUP/jbusecke/projects/cmip_data_management_princeton/catalogs/jupyterrc-zarr-cmip6.json"
        else:
            url = "/projects/GEOCLIM/LRGROUP/jbusecke/projects/cmip_data_management_princeton/catalogs/jupyterrc-cmip6.json"
    elif "jupyter-" in hostname:
        url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        
    # build collection (so that the behavior is the same as in cmip6_pp
    col = intake.open_esm_datastore(url)
    

    if 'time_range' in col.df.columns: # only apply to netcdf catalog
        # I need to apply a more aggressive fix for entries without time_range, otherwise these will not be sorted out properly
        df = col.df.copy(deep=True)
        # split dataframe
        time_range_index = df['time_range'].isnull()
        df_with_time_range = df[~time_range_index]
        df_without_time_range = df[time_range_index]
        df_modified = _pick_latest_version_aggressive(df_without_time_range)
        df_new = pd.concat([df_with_time_range, df_modified])

        col.df = df_new
    
    return col

def o2_models():
    """A central place to store all available models with o2 output"""
    return [
        "ACCESS-ESM1-5",
        "CESM2",
        "CESM2-WACCM",
        "CMCC-ESM2",
        "CNRM-ESM2-1",
        "CanESM5",
        "CanESM5-CanOE",
        "EC-Earth3-CC",
        "GFDL-CM4",
        "GFDL-ESM4",
        "IPSL-CM5A2-INCA",
        "IPSL-CM6A-LR",
        "KIOST-ESM",
        "MIROC-ES2L",
        "MPI-ESM-1-2-HAM",
        "MPI-ESM1-2-HR",
        "MPI-ESM1-2-LR",
        "MRI-ESM2-0",
        "NorESM2-LM",
        "NorESM2-MM",
        "UKESM1-0-LL",
    ]
