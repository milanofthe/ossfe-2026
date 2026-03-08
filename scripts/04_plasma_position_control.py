"""
OSSFE 2026 Tutorial — Example 4: Plasma Vertical Position Control
==================================================================
Simplified model of plasma vertical position stabilization in a tokamak.

The plasma is vertically unstable (positive feedback from elongation).
A PID controller drives the vertical field coils to stabilize position.

Model:
    - Plant: plasma vertical dynamics (unstable 2nd order)
        m_eff * z'' = F_instability + F_coil + F_disturbance
        F_instability = n_index * z  (destabilizing, n_index > 0)
        F_coil = K_coil * I_coil     (stabilizing)
    - Controller: PID on vertical position error
    - Actuator: coil current with first-order lag (L/R time constant)

Run: python 04_plasma_position_control.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Integrator,
    Amplifier,
    Adder,
    Scope,
    Source,
    Constant,
    PID,
    PT1,
)
from pathsim.solvers import RKCK54


# ---- Physical parameters ----

m_eff = 1.0        # effective plasma mass (normalized)
n_index = 2.0      # instability growth rate squared (positive = unstable)
K_coil = 10.0      # coil force per unit current
tau_coil = 0.005   # coil L/R time constant [s] (5 ms)

# PID gains (tuned for stability)
Kp = 50.0
Ki = 10.0
Kd = 5.0

# Disturbance: ELM-like impulse at t=0.1s
def disturbance(t):
    if 0.1 < t < 0.105:
        return 50.0   # impulse force
    elif 0.3 < t < 0.305:
        return -30.0   # another kick
    return 0.0


# ---- Blocks ----

# Reference position (z = 0, centered)
ref = Constant(0.0)

# Position error
err = Adder("+-")

# PID controller
pid = PID(Kp, Ki, Kd, f_max=100)

# Coil actuator: first-order lag
coil = PT1(tau_coil)

# Coil force
coil_force = Amplifier(K_coil)

# Instability force: n_index * z (positive feedback!)
instability = Amplifier(n_index)

# Disturbance source
dist = Source(disturbance)

# Sum of forces on plasma
forces = Adder("++-")  # coil_force + disturbance - instability (note: instability adds, but we model as +n*z pushing away)

# Newton's law: a = F/m
inv_mass = Amplifier(1/m_eff)

# Velocity and position integrators
vel = Integrator(0.0)
pos = Integrator(0.0)

# Scope
sco = Scope(labels=[
    "position z [m]",
    "coil current [A]",
    "disturbance [N]",
])

blocks = [
    ref, err, pid, coil, coil_force,
    instability, dist, forces, inv_mass,
    vel, pos, sco,
]


# ---- Connections ----

connections = [
    # Reference - position = error
    Connection(ref, err[0]),
    Connection(pos, err[1]),

    # Error -> PID -> coil actuator
    Connection(err, pid),
    Connection(pid, coil),

    # Coil current -> force
    Connection(coil, coil_force, sco[1]),

    # Position -> instability force (positive feedback)
    Connection(pos, instability),

    # Sum forces: coil + disturbance - instability
    Connection(coil_force, forces[0]),
    Connection(dist, forces[1], sco[2]),
    Connection(instability, forces[2]),

    # F = ma -> integrate twice
    Connection(forces, inv_mass),
    Connection(inv_mass, vel),
    Connection(vel, pos),
    Connection(pos, sco[0]),
]


# ---- Simulation ----

sim = Simulation(
    blocks,
    connections,
    dt=0.001,
    Solver=RKCK54,
    tolerance_lte_abs=1e-8,
    tolerance_lte_rel=1e-6,
)


if __name__ == "__main__":
    sim.run(duration=0.5)

    sco.plot(lw=2)
    plt.suptitle("Plasma Vertical Position Control")

    plt.show()
