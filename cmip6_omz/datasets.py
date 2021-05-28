# A module to smoothly read in obs data (could probably be handled by an intake catalog...)
# also this should read stuff from relative paths to links in 'external' folder.

#this file is recreated from jbusecke/aguadv_omz_busecke_2021

import xarray as xr
import numpy as np
import warnings

from aguadv_omz_busecke_2021.omz_tools import convert_o2_ml_l

# TODO: We are planning on integrating this functionality in xgcm at a later point. Check back in time to see if we can refactor this.
from xarrayutils.utils import dll_dist


def reconstruct_areacello(ds):
    if "lon" in ds.coords and "lat" in ds.coords:
        if "lon_bounds" in ds.coords and "lat_bounds" in ds.coords:
            # this expects that the preprocessing has converted all the bounds to sorted verticies
            dlon = ds.lon_bounds.diff("bnds").squeeze(drop=True).load()
            # interpolate all points with 0 and negative values from the surrondings.
            dlon = dlon.where(dlon > 0).interpolate_na(dim="x").interpolate_na(dim="y")

            dlat = ds.lat_bounds.diff("bnds").squeeze(drop=True).load()

            # interpolate all points with 0 and negative values from the surrondings.
            dlat = dlon.where(dlat > 0).interpolate_na(dim="y").interpolate_na(dim="x")

            msg = "based on `lon_bounds` and `lat_bounds` difference"
        else:
            warnings.warn(
                "No bounds found for lon and lat. Reconstructing with a very simplified method. Check results carefully."
            )
            lon_dif = ds.lon.data - ds.lon.roll(x=1, roll_coords=False)
            lon_dif = np.where(lon_dif >= 0, lon_dif, lon_dif + 360)
            dlon = xr.DataArray(lon_dif, coords=ds.lon.coords)

            lat_dif = ds.lat.data - ds.lat.roll(y=1, roll_coords=False)
            #             lat_dif = np.where(lat_dif,lat_dif >= 0, 180+ lat_dif)
            dlat = xr.DataArray(lat_dif, coords=ds.lat.coords)
            msg = "based on `lon` and `lat` difference"

        dx, dy = dll_dist(dlon, dlat, ds.lon, ds.lat)
        area = dx * dy
        area.attrs = {"cmip6_preprocessing": {"reconstructed": msg}}
        area.name = "areacello"
        return area.squeeze()
    else:
        warnings.warn(
            "Reconstrution of `areacello` failed. Could not find one of `lon` or `lat` in dataset"
        )
        return None


def correct_lon(ds):
    """Wraps negative x and lon values around to have 0-360 lons.
    longitude names expected to be corrected with `rename_cmip6`"""
    ds = ds.copy()

    x = ds["x"].data
    x = np.where(x < 0, 360 + x, x)

    lon = ds["lon"].where(ds["lon"] > 0, 360 + ds["lon"])

    ds = ds.assign_coords(x=x, lon=(ds.lon.dims, lon))
    ds = ds.sortby("x")
    return ds


#
# WOA stuff
#


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

    # I usually use aou as negative
    attrs = woa["aou"].attrs
    woa["aou"] = -woa["aou"]
    woa["aou"].attrs = attrs

    # recalculate o2sat
    woa["o2sat"] = woa["o2"] - woa["aou"]
    woa["o2sat"].attrs["units"] = woa["aou"].attrs["units"]

    # # reconstruct area
    woa.coords["areacello"] = reconstruct_areacello(woa)
    # this is a shitty old implementation (xr cf will solve this....)
    woa.coords["areacello"].attrs = {}

    woa = woa.assign_coords(
        thkcello=("lev", woa.lev_bounds.diff("bnds").squeeze().data)
    )

    return woa
