# Run-and-Tumble Particle Potentials

This repository contains the mathematical framework, numerical simulations, and analysis tools for studying the rectification of **Run-and-Tumble (RnT) particles** in asymmetric ratchet potentials.

## 📌 Overview

Run-and-Tumble particles are a fundamental model in active matter physics. Unlike passive Brownian particles, RnT particles exhibit self-propulsion interspersed with random reorientation events ("tumbles"). Due to this intrinsic non-equilibrium behavior, their undirected motion can be rectified by asymmetric periodic potentials to generate a steady net particle current.

Using **perturbative field theory**, this project calculates the theoretical potential profiles that maximize particle throughput as a function of key physical parameters (such as propulsion speed, tumble rate, and spatial constraints).

---

## 🚀 Key Features

- **Field-Theoretic Solver:** Analytical approximations for RnT particle probability distributions and current density using perturbative expansions.
- **Optimal Potential Finder:** Algorithms to compute the exact potential shape $V(x)$ that yields maximum rectification current.
- **Stochastic Simulations:** Numerical Langevin/Monte Carlo simulations of individual RnT particles to validate field-theoretic predictions.
- **Parameter Sweeps:** Tools to analyze current efficiency across different propulsion velocities, persistence lengths, and potential amplitudes.

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.8+**
- Core scientific Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`

### Installation

Clone the repository and set up the environment:

```bash
git clone [https://github.com/ks-1019/RnT-Particle-Potentials.git](https://github.com/ks-1019/RnT-Particle-Potentials.git)
cd RnT-Particle-Potentials

# Install dependencies
pip install -r requirements.txt
