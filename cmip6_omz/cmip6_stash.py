def cmip6_dataset_id(ds, sep="_"):
    """Creates a unique string id for e.g. saving files to disk from CMIP6 output"""
    id_components = [
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "table_id",
        "grid_label",
        "version",
    ]
    dataset_id = sep.join(
        [ds.attrs[i] if i in ds.attrs.keys() else "none" for i in id_components]
    )
    return dataset_id
