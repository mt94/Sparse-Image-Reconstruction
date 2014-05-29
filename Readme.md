# Readme

## Overview
The source code was used to generate the results of the work *Molecular Imaging in Nano MRI* by Michael Ting, Wiley 2014. The latest version is 0.0.1. 

## Licence
It is released under the GNU General Public Licence (GPL) version 3. Please refer to the terms and conditions at [https://gnu.org/licenses/gpl.html](https://gnu.org/licenses/gpl.html).

## Software requirements
The source code was developed in Python 2.7.3, and makes use of the packages Numpy 1.8.0, Scipy 0.13.0, Pymc 2.2, Matplotlib 1.2.0.

## Getting started
Looking in the `Examples` subdirectory is instructive for how to use the various classes. The second step would be to look through the classes in the `Channel`, `Recon`, `Sim`, and `Systems` subdirectories to extend or modify them. Finally, the `SimRuns` subdirectory contains files that use the `Examples` subdirectory to generate simulation results.