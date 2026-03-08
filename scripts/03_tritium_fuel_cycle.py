"""
OSSFE 2026 Tutorial — Example 3: Tritium Fuel Cycle
=====================================================
A simplified tritium fuel cycle model for a fusion reactor,
using pathsim-chem's tritium modeling blocks.

Compartments:
    - Plasma:     tritium burned (D-T fusion) + unburned exhaust
    - Blanket:    breeds tritium via neutron capture in lithium (TBR > 1)
    - Exhaust:    processes unburned fuel for recovery
    - Storage:    central tritium inventory

Uses ResidenceTime blocks from pathsim-chem to model first-order
mass transport with characteristic residence times.

Install: pip install pathsim pathsim-chem

Run: python 03_tritium_fuel_cycle.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Adder, Scope, Constant
from pathsim.solvers import RKCK54

from pathsim_chem.tritium import ResidenceTime, Splitter


# ---- Fuel cycle parameters ----

burn_fraction = 0.02      # 2% of injected fuel is burned (typical for tokamaks)
TBR = 1.05                # tritium breeding ratio (must be > 1 for self-sufficiency)
fueling_rate = 1.0        # g/s tritium injection rate
exhaust_recovery = 0.99   # 99% recovery from exhaust processing

tau_exhaust = 3600         # exhaust processing residence time [s] (1 hour)
tau_blanket = 1800         # blanket extraction residence time [s] (30 min)
tau_storage = 1e12         # storage "residence time" (effectively infinite — no decay)


# ---- Blocks ----

# Fueling source
fuel_in = Constant(fueling_rate)

# Plasma: split fuel into burned and unburned fractions
plasma_split = Splitter(fractions=[burn_fraction, 1 - burn_fraction])

# Breeding: TBR * burn_rate gives tritium production in blanket
breeding = Amplifier(TBR)

# Blanket: tritium extraction with residence time
blanket = ResidenceTime(tau=tau_blanket, initial_value=0)

# Exhaust processing: unburned fuel recovery with residence time
exhaust = ResidenceTime(tau=tau_exhaust, initial_value=0)

# Recovery efficiency on exhaust output
recovery = Amplifier(exhaust_recovery)

# Storage inventory: accumulates recovered + bred - fueling
storage = ResidenceTime(
    tau=tau_storage,
    betas=[1, 1, -1],      # +recovered, +bred, -fueling
    initial_value=100.0,    # start with 100g initial inventory
)

# Scope to track key quantities
sco = Scope(labels=[
    "storage [g]",
    "exhaust [g]",
    "blanket [g]",
    "burn rate [g/s]",
])

blocks = [
    fuel_in, plasma_split, breeding,
    blanket, exhaust, recovery, storage, sco,
]


# ---- Connections ----

connections = [
    # Fuel -> plasma split (burned / unburned)
    Connection(fuel_in, plasma_split, storage[2]),

    # Burned fraction -> breeding -> blanket
    Connection(plasma_split[0], breeding, sco[3]),
    Connection(breeding, blanket),

    # Unburned fraction -> exhaust processing
    Connection(plasma_split[1], exhaust),

    # Exhaust output (flux) -> recovery -> storage
    Connection(exhaust["x/tau"], recovery),
    Connection(recovery, storage[0]),

    # Blanket output (flux) -> storage
    Connection(blanket["x/tau"], storage[1]),

    # Scope connections
    Connection(storage, sco[0]),
    Connection(exhaust, sco[1]),
    Connection(blanket, sco[2]),
]


# ---- Simulation ----

sim = Simulation(
    blocks,
    connections,
    dt=10,
    Solver=RKCK54,
)


if __name__ == "__main__":
    # Simulate 24 hours
    sim.run(duration=86400)

    sco.plot(lw=2)
    plt.suptitle("Tritium Fuel Cycle — 24h Simulation")

    plt.show()
