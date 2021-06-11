# A module to smoothly read in obs data (could probably be handled by an intake catalog...)
# also this should read stuff from relative paths to links in 'external' folder.

import xarray as xr
import numpy as np
import warnings

from cmip6_preprocessing.preprocessing import correct_lon

from omz_tools import convert_o2_ml_l
from upstream_stash import recontruct_areacello


### WOA stuff ###

def preprocess(woa):
    woa = woa.copy()
    woa = woa.rename(
        {
            "lon": "x",
            "lat": "y",
            "lon_bnds": "lon_bounds",
            "lat_bnds": "lat_bounds",
            "depth_bnds": "lev_bounds",
            "depth": "lev",
            "nbounds": "bnds",
        }
    )
    woa.coords["lon"] = woa.x * xr.ones_like(woa.y)
    woa.coords["lat"] = xr.ones_like(woa.x) * woa.y
    woa = woa.assign_coords(
        {
            k: woa[k]
            for k in [
                "crs",
                "climatology_bounds",
                "lev_bounds",
            ]
        }
    )
    woa = correct_lon(woa)
    # there is some funkyness with the bounds. we dont need it here
    woa = woa.drop_vars(["lon_bounds", "lat_bounds"])
    return woa


def woa13():
    # woa aou
    woa_filelist = [
        "/tigress/GEOCLIM/LRGROUP/shared_data/salt_WOA/woa13_decav_s00_04v2.nc",
        "/tigress/GEOCLIM/LRGROUP/shared_data/temp_WOA/woa13_decav_t00_04v2.nc",
        "/tigress/GEOCLIM/LRGROUP/shared_data/oxygen_WOA/woa13_all_o00_01.nc",
        "/tigress/GEOCLIM/LRGROUP/shared_data/aou_WOA/woa13_all_A00_01.nc",
        "/tigress/GEOCLIM/LRGROUP/shared_data/phosphate_WOA/woa13_all_p00_01.nc",
        "/tigress/GEOCLIM/LRGROUP/shared_data/nitrate_WOA/woa13_all_n00_01.nc",
    ]
    datasets = [
        xr.open_dataset(file, decode_times=False, chunks={"depth": 1}).squeeze()  #
        for file in woa_filelist
    ]

    datasets = [preprocess(ds) for ds in datasets]
    datasets[5] = datasets[5].interp_like(datasets[0].isel(lev=0).squeeze())
    datasets[4] = datasets[4].interp_like(datasets[0].isel(lev=0).squeeze())
    datasets[3] = datasets[3].interp_like(datasets[0].isel(lev=0).squeeze())
    datasets[2] = datasets[2].interp_like(datasets[0].isel(lev=0).squeeze())

    woa = xr.merge(datasets, compat="override")
    woa = woa.drop_vars([v for v in woa.data_vars if "an" not in v])
    woa = woa.rename(
        {
            "o_an": "o2",
            "A_an": "aou",
            "t_an": "thetao",
            "s_an": "so",
            "p_an": "po4",
            "n_an": "no3",
        }
    )
    #!!! this is very likely to break
    woa.no3.attrs = datasets[5].n_an.attrs
    woa.po4.attrs = datasets[4].p_an.attrs
    woa.o2.attrs = datasets[2].o_an.attrs
    woa.aou.attrs = datasets[3].A_an.attrs
    woa.so.attrs = datasets[0].s_an.attrs
    woa.thetao.attrs = datasets[1].t_an.attrs

    # convert oxygen units into mol/m^3
    for va in ["o2", "aou"]:
        attrs = woa[va].attrs
        woa[va] = (
            convert_o2_ml_l(woa[va]) * 1025
        )  # should convert from mol/kg to mol/m^3
        attrs["units"] = "$mol \, m^3$"
        woa[va].attrs = attrs

    # convert nutrient units into mol/m^3
    for va in ["po4", "no3"]:
        attrs = woa[va].attrs
        woa[va] = woa[va] / 1000
        attrs["units"] = "$mol \, m^3$"
        woa[va].attrs = attrs

    # recalculate o2sat
    woa["o2sat"] = woa["o2"] - woa["aou"]
    woa["o2sat"].attrs["units"] = woa["aou"].attrs["units"]

    # reconstruct area
    woa.coords["areacello"] = reconstruct_areacello(woa)
    woa.coords["areacello"].attrs = {}

    woa = woa.assign_coords(
        thkcello=("lev", woa.lev_bounds.diff("bnds").squeeze().data)
    )

    return woa
