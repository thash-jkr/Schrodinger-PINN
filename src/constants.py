from scipy.constants import pi, speed_of_light, elementary_charge, electron_mass, hbar
import numpy as np

# Change of units
me_SI = electron_mass           # Electron mass in kg
hbar_SI = hbar                  # Reduced Planck constant in J·s
e_SI = elementary_charge        # Elementary charge in C
c_SI = speed_of_light           # Speed of light in m/s
h_SI = hbar_SI * 2 * pi         # Planck constant in J·s

meV = e_SI * 1e-3               # 1 meV in Joules
nm = 1e-9                       # 1 nm in meters
ps = 1e-12                      # 1 ps in seconds

c = c_SI * ps / nm              # Speed of light in nm/ps (~299792.458 nm/ps)
e = 1e3                         # 1 eV = 1e3 meV
hbar_meV_ps = hbar_SI / (meV * ps)   # hbar in meV·ps
h_bar_meV_ps = hbar_meV_ps           # Alias
me = me_SI * c_SI**2 / meV / c**2    # Electron mass in meV·ps²/nm²

# Semiconductors
mt = 0.19 * me    # Electron transverse effective mass in meV·ps²/nm²
ml = 0.92 * me    # Electron longitudinal effective mass in meV·ps²/nm²

boykin_unit_cell_length = 0.2716  # Unit cell length in nm

# Magnonics
gyromagnetic_ratio = 2 * elementary_charge / (2 * electron_mass)  # rad·s⁻¹·T⁻¹
