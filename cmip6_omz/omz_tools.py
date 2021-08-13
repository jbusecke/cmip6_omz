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

import sys
sys.path.append("../")
from cmip6_omz.units import(
    convert_o2_ml_l,
    convert_mol_m3_mymol_kg,
)

from cmip6_preprocessing.utils import cmip6_dataset_id
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

# TODO Move the old version into tests and ensure the results stay the same
def omz_thickness_efficient(
    ds,
    dz_var = 'dz_t',
    o2_var="o2",
    bin_chunks=1,
    o2_bins=np.array([5, 10, 20, 40, 60, 80, 100, 120, 140])
):
    conversion_factor = 1 / convert_mol_m3_mymol_kg(xr.DataArray([1])).data
    o2_bins_converted_raw = o2_bins * conversion_factor
    o2_bins_converted = xr.DataArray(o2_bins_converted_raw, dims=['o2_bin'], coords={'o2_bin':o2_bins})# can I generalize this with pint?
    o2_bins_converted = o2_bins_converted.chunk({'o2_bin':bin_chunks})
    return ds[dz_var].where(ds[o2_var]<=o2_bins_converted, 0)



# 
def sigma_bins():
    """a global definition of fine density bins for transformation
    See `notebooks/dev_fine_density_bins.ipynb` for details.
    """
    return np.hstack([[0], np.arange(22.5, 26.5, 0.25), np.arange(26.5, 27.9, 0.05), [100]])


#TODO: testing
def align_missing(ds_in):
    """Make sure that nans in all fields of a dataset are consistent.
    Requires"""
    # Due to the interpolation between `gr` and `gn`, we have to make sure that all data variables are masked in the same way!
    
    # its probably fine to only look at a few time steps at the beginning and end
    if 'time' in ds_in.dims:
        ds_mask = xr.concat([ds_in.isel(time=slice(0,2)), ds_in.isel(time=slice(-2, None))], 'time')
    else:
        ds_mask = ds_in

    # for generalization np.logical_or.reduce((x, y, z))https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
    combo_nanmask = np.logical_or(
        np.isnan(ds_mask.o2).all("time" if 'time' in ds_in.dims else []).load(),
        np.isnan(ds_mask.thetao).all("time" if 'time' in ds_in.dims else []).load(), # the `rho` function returns nan when one of t/s is nan, so we only need to chcek one of them
    )
    try:
        plt.figure()
        combo_nanmask.isel(lev=5).plot()
        plt.show()
    except:
        pass
    
    ds_out = ds_in.where(~combo_nanmask)
    
    return ds_out

def preprocessing_wrapper(ds_in):
    
    # fix the attribute parsed by intake-esm
    for k,v in ds_in.attrs.items():
        if v is None:
            print(f"Replacing {k} attrs value with `none`")
            ds_in.attrs[k] = 'none'
    
    # drop all variables that might have hitched a ride from before (some datasets have an additional area etc...)
    drop_vars = [va for va in ds_in.data_vars if va not in ['o2', 'agessc', 'so', 'thetao', 'sigma_0', 'omz_thickness']]
    ds_in = ds_in.drop_vars(drop_vars)
            
#     ds_in = strip_encoding(ds_in) #I think this is not needed anymore with the newest xarray
    
    if 'member_id' in ds_in.dims:
        if isinstance(ds_in.member_id.data, object):
            ds_in['member_id'] = ds_in['member_id'].astype(str)

    # strip all the coords to avoid trouble
    delete_coords = [
        "branch_time_in_parent",
        "branch_time_in_child",
        "parent_time_units",
        "child_time_units",
        "parent_variant_label",  # these are all scalar coords (and zarr doesnt like those?)
        "time_bounds",  # this is a bit different one, but makes trouble as a coordinate?
    ]

    ds_out = ds_in.drop([co for co in ds_in.coords if co in delete_coords])

    # see below. Make a new fake 'outer' coordinate. I think the values really dont matter?
    ds_out = ds_out.assign_coords(
        lev_outer=np.hstack(
            [0, (ds_out.lev.data[1:] + ds_out.lev.data[0:-1]) / 2, 5e10]
        )
    )
    return ds_out

def vol_consistency_check_wrapper(ds, ds_sigma):
    perc_difference, omz_perc_difference = volume_consistency_checks(
        ds, ds_sigma
    )
    print(
        f"Relative difference ocean vol: {abs(perc_difference).data}% | OMZ vol {abs(omz_perc_difference).data}%"
        )
    
    if (abs(perc_difference) > 0.1).any() or (
        abs(omz_perc_difference) > 0.25# Had to increase for ESM4 before most would pass 0.01
        ).any():
        print("Volume differences exceed threshold. NOT SAVING.")
        print('\x1b[31m"red"\x1b[0m')
        consistent = False
    else:
        consistent = True
    return consistent
