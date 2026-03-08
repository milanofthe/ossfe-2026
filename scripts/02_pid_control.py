"""
OSSFE 2026 Tutorial — Example 2: PID Controller
=================================================
A PID controller tracking a setpoint through an integrating plant.

Classic control loop:
    setpoint -> [error] -> [PID] -> [plant] -> output
                  ^                              |
                  |______________________________|

Run: python 02_pid_control.py
"""

import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope, PID
from pathsim.solvers import RKCK54


# Plant gain
K = 0.4

# PID parameters
Kp = 1.5   # proportional
Ki = 0.5   # integral
Kd = 0.1   # derivative

# Setpoint: step changes at t=20 and t=60
def setpoint(t):
    if t > 60:
        return 0.5
    elif t > 20:
        return 1.0
    else:
        return 0.0


# Blocks
src = Source(setpoint)
err = Adder("+-")                  # error = setpoint - output
pid = PID(Kp, Ki, Kd, f_max=10)   # PID with output saturation
pnt = Integrator()                 # integrating plant
pgn = Amplifier(K)                 # plant gain
sco = Scope(labels=["setpoint", "output", "error"])

blocks = [src, err, pid, pnt, pgn, sco]

# Connections
connections = [
    Connection(src, err, sco[0]),      # setpoint -> error, scope
    Connection(pgn, err[1], sco[1]),   # output -> error (feedback), scope
    Connection(err, pid, sco[2]),      # error -> PID, scope
    Connection(pid, pnt),              # PID output -> plant
    Connection(pnt, pgn),              # plant state -> plant gain
]

# Simulation
sim = Simulation(blocks, connections, Solver=RKCK54)


if __name__ == "__main__":
    sim.run(100)

    sco.plot(lw=2)

    plt.show()
