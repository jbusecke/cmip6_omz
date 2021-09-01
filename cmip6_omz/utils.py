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
    elif "jupyter.rc" in hostname:
        # testing if this works on jupyterrc?
        if zarr:
            url = "/projects/GEOCLIM/LRGROUP/jbusecke/projects/cmip_data_management_princeton/catalogs/jupyterrc-zarr-cmip6.json"
        else:
            url = "/projects/GEOCLIM/LRGROUP/jbusecke/projects/cmip_data_management_princeton/catalogs/jupyterrc-cmip6.json"
    elif "jupyter-" in hostname:
        url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    return url

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
