import xarray as xr
from cmip6_omz.cmip6_stash import cmip6_dataset_id

def test_cmip6_dataset_id():
    ds = xr.Dataset()
    ds.attrs = {
        "activity_id": "a",
        "institution_id": "b",
        "source_id": "c",
        "experiment_id": "d",
        "table_id": "e",
        "grid_label": "f",
        "version": "v2",
    }
    assert cmip6_dataset_id(ds) == "a_b_c_d_e_f_v2"
    assert cmip6_dataset_id(ds, sep=".") == "a.b.c.d.e.f.v2"
    # test case with one missing
    del ds.attrs["source_id"]
    assert cmip6_dataset_id(ds) == "a_b_none_d_e_f_v2"
