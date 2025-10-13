# RuptureGenerator
The main script, RuptG.py, contains all core functions and routines required to generate rupture models.

To demonstrate its functionality, several example scripts are provided:

  Example1.py – Illustrates how to create a single rupture realization in a sequential process.

  Example2.py – Demonstrates how to generate multiple rupture realizations in parallel using functions from RuptG.py.

  Example3.py – Shows how to enhance the high-frequency content of an initial rupture description to produce multiple realizations.

Visualization

Two scripts are provided to visualize the generated ruptures:

  LectorFuente.py – Plots the main features of each generated rupture. By default, it visualizes results from Example1.py.

  LectorFuenteTot.py – Plots features of all ruptures generated in parallel (default: results from Example2.py).

Ground Motion Simulation

  The S_AxitraMult.py script runs the generated ruptures using Axitra to compute ground motions at multiple locations.


We are glad if you use this code or methodology in your research, please cite the following publication:
David Castro-Cruz, Paul Martin Mai, A new kinematic rupture generation technique and its application, Geophysical Journal International, Volume 243, Issue 3, December 2025, ggaf385, https://doi.org/10.1093/gji/ggaf385 

