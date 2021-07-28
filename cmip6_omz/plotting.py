import matplotlib.pyplot as plt
from cmip6_preprocessing.utils import cmip6_dataset_id


def plot_omz_results(ds):
    dataset_id = cmip6_dataset_id(ds)
    
#     da = mask_pacific(ds)
    ds = ds.where(abs(ds.lat) <= 30, drop=True)

    area = ds.areacello.fillna(0)

    plt.figure()
    ds.thetao.isel(time=0, sigma_0=2).plot(robust=True, x="lon", y="lat")
    plt.title(dataset_id)

    plt.figure()
    vol = (ds.omz_thickness * area).sum(["x", "y"])
    vol = vol.sel(o2_bin=80)
    (vol - vol.mean("time")).plot(hue="sigma_0")
    plt.ylabel("omz vol")
    plt.title(dataset_id)

    plt.figure()
    o2 = (ds.o2).weighted(area).mean(["x", "y"])
    (o2 - o2.mean("time")).plot(hue="sigma_0")
    plt.ylabel("mean o2")
    plt.title(dataset_id)

    if "agessc" in ds.data_vars:
        plt.figure()
        age = (ds.agessc).weighted(area).mean(["x", "y"])
        (age - age.mean("time")).plot(hue="sigma_0")
        plt.ylabel("mean age")
        plt.title(dataset_id)