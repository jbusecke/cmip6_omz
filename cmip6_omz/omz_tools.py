# Tools for omz detection and processing
try:
    import gsw
except:
    gsw = None
    
try:
    import regionmask
except:
    regionmask = None

import numpy as np
import warnings
import xarray as xr
#import matplotlib.pyplot as plt

from .units import(
    convert_o2_ml_l,
    convert_mol_m3_mymol_kg,
)

from aguadv_omz_busecke_2021.cmip6_stash import cmip6_dataset_id
from cmip6_preprocessing.regionmask import merged_mask


###Mask Function###
def mask_basin(ds, region='Pacific', drop=True):
    if regionmask is None:
        raise RuntimeError("Please install the latest regionmask version")
    basins = regionmask.defined_regions.natural_earth.ocean_basins_50
    mask = merged_mask(basins, ds)
    masks = {
        'Pacific':np.logical_or(mask == 2, mask == 3),
        'Atlantic':np.logical_or(mask == 0, mask == 1),
        'Indian':mask == 5,  # Indian without Maritime Continent
        'Global': mask >= 0
    }
    ds_masked = ds.where(masks[region], drop=drop)
    return ds_masked

###Volume Functions###

def sample_select(ds_check):
    if 'member_id' in ds_check.dims:
        ds_check = ds_check.isel(member_id=0)
    ds_check = ds_check.isel(time=slice(12,48))
    return ds_check

def full_volume(ds_check):
    ds_check = sample_select(ds_check)
    
    if 'areacello' in ds_check.variables:
        area = ds_check.areacello
    else:
        area = ds_check.dx_t * ds_check.dy_t
    dz = xr.ones_like(ds_check.o2) * ds_check.dz_t
    vol = (dz * area)
    # mask out nans correctly
    if len(ds_check.dz_t.shape) == 1:
        vol = vol.where(~np.isnan(ds_check.sigma_0))
    return vol

def omz_full_volume(ds_check):
    ds_check = sample_select(ds_check)
    
    if 'areacello' in ds_check.variables:
        area = ds_check.areacello
    else:
        area = ds_check.dx_t * ds_check.dy_t
    
    vol = (ds_check.omz_thickness * area)
    return vol

def volume_consistency_checks(ds_z, ds_sigma):
    print("Check if ocean volume is conserved...")
    vol_pre = full_volume(ds_z).sum(["x", "y", "lev"])
    vol_post = full_volume(ds_sigma).sum(["x", "y", "sigma_0"])
    vol_perc_difference = (vol_post - vol_pre) / vol_pre * 100
    vol_perc_difference = vol_perc_difference.mean("time").load()

    omz_vol_pre = omz_full_volume(ds_z).sum(["lev", "x", "y"])
    omz_vol_post = omz_full_volume(ds_sigma).sum(
        ["sigma_0", "x", "y"]
    )
    omz_perc_difference = (omz_vol_post - omz_vol_pre) / omz_vol_pre * 100
    omz_perc_difference = omz_perc_difference.mean("time").load()
    return vol_perc_difference, omz_perc_difference




#Compute OMZ thickness
def omz_thickness(
    ds,
    dz_var="dz_t",
    o2_var="o2",
    o2_bins=np.array([5, 10, 20, 40, 60, 80, 100, 120, 140]),
):
    #should this check units of o2?
    o2 = ds[o2_var] / 1025 * 1e6 #from mol/m^3 to mymol/kg
    dz = xr.ones_like(o2) * ds[dz_var]
    datasets = [
        dz.where(o2 <= o2b, 0).assign_coords(o2_bin=o2b).astype(o2.dtype)
        for o2b in o2_bins
    ]
    return xr.concat(datasets, dim="o2_bin")








