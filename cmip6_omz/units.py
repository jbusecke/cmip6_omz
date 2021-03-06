###Conversion functions###

def convert_o2_ml_l(o2):
    """Convert oxygen concentrations in ml/l to SI units (mol/kg).
    https://www.nodc.noaa.gov/OC5/WOD/wod18-notes.html"""
    o2_mol_vol = 22.392  # l/mole #what is this doing?
    # this must be used implicitly in the above?
    # , rho_0=1025
    converted = o2 * 43.570 / 1e6
    converted.attrs["units"] = "$mol/kg$"
    return converted

def convert_mol_m3_mymol_kg(o2, rho_0=1025):
    converted = o2 / rho_0 * 1e6
    converted.attrs["units"] = "$\mu mol/kg$"
    return converted

def convert_mg_l_to_mymol_kg(o2, rho_0=1025):
    """Convert oxygen concentrations in ml/l to mymol/kg."""
    converted = o2 * 1/32000 * rho_0/1000 * 1e6
    converted.attrs["units"] = "$\mu mol/kg$"
    return converted
