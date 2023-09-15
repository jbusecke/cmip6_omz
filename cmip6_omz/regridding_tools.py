import xarray as xr
import xesmf as xe


def regrid_regular(ds, spacing=0.5):
    target_grid = xe.util.grid_global(spacing, spacing)
    # this makes problems whan passing a dataarray.
    # find a better way to drop everythig but lon/lat
    regridder = xe.Regridder(
        ds.drop([v for v in ds.variables if v not in ["lon", "lat"]]),
        target_grid,
        "bilinear",
        periodic=True,
        ignore_degenerate=True,
    )
    return regridder(ds)
