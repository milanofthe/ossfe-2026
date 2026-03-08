"""
OSSFE 2026 Tutorial — Example 1: Harmonic Oscillator
=====================================================
A spring-mass-damper system modeled with basic blocks.

This is the "Hello World" of PathSim: two integrators (velocity, position),
amplifiers for spring and damping forces, and an adder to sum forces.

    m*x'' + d*x' + k*x = 0

Run: python 01_harmonic_oscillator.py
"""

import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Adder, Scope
from pathsim.solvers import SSPRK33


# Parameters
m = 0.8    # mass [kg]
d = 0.2    # damping [Ns/m]
k = 1.5    # spring constant [N/m]

# Initial conditions
x0 = 2.0   # initial position [m]
v0 = 5.0   # initial velocity [m/s]


# Blocks
vel = Integrator(v0)       # velocity integrator
pos = Integrator(x0)       # position integrator
dmp = Amplifier(d)         # damping force
spr = Amplifier(k)         # spring force
inv = Amplifier(-1/m)      # Newton's 2nd law: a = -F/m
add = Adder()              # sum of forces
sco = Scope(labels=["velocity", "position"])

blocks = [vel, pos, dmp, spr, inv, add, sco]

# Connections — this IS the block diagram
connections = [
    Connection(vel, pos, dmp, sco),     # velocity -> position, damping, scope
    Connection(pos, spr, sco[1]),       # position -> spring, scope
    Connection(dmp, add),               # damping -> adder
    Connection(spr, add[1]),            # spring -> adder
    Connection(add, inv),               # total force -> 1/m
    Connection(inv, vel),               # acceleration -> velocity (feedback)
]

# Simulation
sim = Simulation(blocks, connections, dt=0.1, Solver=SSPRK33)


if __name__ == "__main__":
    sim.run(duration=30)

    sco.plot(lw=2)
    sco.plot2D(lw=2)  # phase portrait

    plt.show()
