# PathSim Tutorial — OSSFE 2026

Tutorial materials for the [OSSFE 2026](https://ossfe.org/OSSFE_2026/) conference in Munich (June 10–12, 2026).

**Session:** [T7 — PathSim: An Open-Source Python Framework for Dynamic System Simulation in Fusion Energy Applications](https://ossfe.org/OSSFE_2026/programme/abstracts/rother-pathsim/)

**Author:** Milan Rother


## Interactive Examples (Documentation)

These examples run directly in the browser on the documentation site — no installation needed:

### PathSim Basics

| Example | Description |
|---------|-------------|
| [Linear Feedback](https://docs.pathsim.org/pathsim/v0.18.0/examples/linear-feedback) | First-order linear feedback — the simplest PathSim model |
| [Harmonic Oscillator](https://docs.pathsim.org/pathsim/v0.18.0/examples/harmonic-oscillator) | Damped spring-mass-damper system |
| [PID Controller](https://docs.pathsim.org/pathsim/v0.18.0/examples/pid-controller) | Classical PID feedback control |
| [Chemical Reactor](https://docs.pathsim.org/pathsim/v0.18.0/examples/chemical-reactor) | Chemical reaction kinetics with nonlinear dynamics |

### PathSim-Chem Toolbox

| Example | Description |
|---------|-------------|
| [Point Kinetics](https://docs.pathsim.org/chem/v0.2.2/examples/point-kinetics) | Neutron point kinetics with delayed neutron groups |
| [CSTR Reaction](https://docs.pathsim.org/chem/v0.2.2/examples/cstr-reaction) | Continuous stirred-tank reactor dynamics |
| [Heat Exchanger](https://docs.pathsim.org/chem/v0.2.2/examples/heat-exchanger) | Counter-current heat exchanger |
| [Process Flowsheet](https://docs.pathsim.org/chem/v0.2.2/examples/process-flowsheet) | Multi-unit process simulation |

## Fusion Showcase — PathView Models

Open these directly in [PathView](https://view.pathsim.org) — no installation needed:

### Core Fusion Models

| Model | Open in PathView |
|-------|-----------------|
| Tritium Fuel Cycle | [Open](https://view.pathsim.org/?model=https://raw.githubusercontent.com/milanofthe/ossfe-2026-tutorial/main/models/tritium-fuel-cycle.pvm) |
| Plasma Position Control | [Open](https://view.pathsim.org/?model=https://raw.githubusercontent.com/milanofthe/ossfe-2026-tutorial/main/models/plasma-position-control.pvm) |
| Point Kinetics | [Open](https://view.pathsim.org/?model=https://raw.githubusercontent.com/milanofthe/ossfe-2026-tutorial/main/models/point-kinetics.pvm) |
| Plasma Power Balance | [Open](https://view.pathsim.org/?model=https://raw.githubusercontent.com/milanofthe/ossfe-2026-tutorial/main/models/plasma-power-balance.pvm) |

### ARC Fusion Reactor Fuel Cycle

Tritium fuel cycle models of the ARC reactor (MIT PSFC):

| Model | Open in PathView |
|-------|-----------------|
| ARC Single BCR | [Open](https://view.pathsim.org/?model=https://raw.githubusercontent.com/milanofthe/ossfe-2026-tutorial/main/models/arc_single_bcr.pvm) |
| ARC (Meschini config) | [Open](https://view.pathsim.org/?model=https://raw.githubusercontent.com/milanofthe/ossfe-2026-tutorial/main/models/arc_same_as_meschini.pvm) |

## Links

- [PathSim GitHub](https://github.com/milanofthe/pathsim) — simulation framework
- [PathView](https://view.pathsim.org) — browser-based graphical editor
- [Documentation](https://docs.pathsim.org)
- [JOSS Paper](https://doi.org/10.21105/joss.07484)

## License

MIT
