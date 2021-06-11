import xarray as xr
import warnings
import numpy as np

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

            #sjd I don't think rolling makes sense for non-periodic coord
            #lat_dif = ds.lat.data - ds.lat.roll(y=1, roll_coords=False)
            #lat_dif = np.where(lat_dif >= 0, lat_diff + 180, lat_dif)
            lat_dif = np.empty_like(ds.lat.data)
            lat_dif[1:] = ds.lat.data[1:]-ds.lat.data[:-1]
            #approximate nominal spacing by taking mean of interior points
            #in order to fill in edge value
            lat_dif[0] = np.nanmean(lat_dif[1:])
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
