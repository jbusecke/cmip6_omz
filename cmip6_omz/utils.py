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
            url = "/tigress/GEOCLIM/LRGROUP/jbusecke/projects/cmip_data_management_princeton/catalogs/jupyterrc-zarr-cmip6.json"
        else:
            url = "/tigress/GEOCLIM/LRGROUP/jbusecke/projects/cmip_data_management_princeton/catalogs/jupyterrc-cmip6.json"
    elif "jupyter-" in hostname:
        url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    return url

def o2_models():
    """A central place to store all available models with o2 output"""
    return [
        "CanESM5-CanOE",
        "CanESM5",
        "CNRM-ESM2-1",
        "ACCESS-ESM1-5",
        "MPI-ESM-1-2-HAM",
        "IPSL-CM6A-LR",
        "MIROC-ES2L",
        "UKESM1-0-LL",
        "MPI-ESM1-2-HR",
        "MPI-ESM1-2-LR",
        "MRI-ESM2-0",
        "NorCPM1",
        "NorESM1-F",
        "NorESM2-LM",
        "NorESM2-MM",
        "GFDL-CM4",
        "GFDL-ESM4",
    ]
