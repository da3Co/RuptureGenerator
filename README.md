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

-------------------------------------------------------------------------------------------------------------------------------------------------------------
Structure of the Object Rupture:

Each object Rupture, coded in RupG.py possesses several attributes to define the rupture process. The main values are:

Ruptures.LocalMo: it stores the local moment source time function at each sub fault. For optimization, this variable store lists on the format (nl, nw). nl and nw are the number of subfaults in the strike direction and in the dip direction respectively. Each sub fault spot contains None if in the sub fault has slip of zero. If the rupture activates the sub fault, inside there are two lists, one containing the index of the time when the subfault starts (tii) and when the subfault finishes (tff). The second list contains the vector of the moment source time function between the times Rupture.timiT[tii:tff]
Rupture.timiT: Time vector for the entire fault.
Rupture.Slip: A np.array containing the slip values at each sub fault (nl, nw)
Ruptures.Trise: A np.array containing Rise time values at each sub fault (nl, nw)
Ruptures.Vr: A np.array containing Rupture velocity at each sub fault (nl, nw)
Ruptures.To: A np.array containing Onset time values at each sub fault (nl, nw)
Ruptures.rakes: A np.array containing  Rakes values at each sub fault (nl, nw)
Rupture.Pos: Matrix showing in metric coordinates the location of each sub fault, first dimension indicates the coordinates x, y, and z. The second and third dimensions select the location of the sub fault in the plane (3, nl, nw)

Rupture.Mo: Seismic moment
Rupture.Mw: Moment magnitude
Rupture.L: Length in strike direction
Rupture.W: width in the dip direction
Rupture.dll: size of the sub fault in the strike direction
Rupture.dww: size of the sub fault in the dip direction
Rupture.strike: strike  in Degrees
Rupture.dip: Dip in Degrees
Rupture.Arake: Average rake in Degrees
Ruptures.chyo: index of the hypocenter

