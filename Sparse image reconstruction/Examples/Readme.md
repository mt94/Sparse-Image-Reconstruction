# Examples Readme

## Introduction
The Examples directory contains the source files used to generate the examples in the work *Molecular Imaging in Nano MRI*. These are the files with classes that derive from `AbstractReconstructorExample`, and are listed in the next section. The second group of files contains classes that derive from `AbstractBlurWithNoise`, e.g., Mrfm2dBlurWithNoise.py. These do not use a reconstructor, but simply apply the blur operation and inject noise into the output. Lastly, a third group of files contains classes that derive from `AbstractExample`, the most basic abstract class of this directory (NB. the first two abstract example classes descend from `AbstractExample`). These files demonstrate lower-level functionality, e.g., in TruncateStandardNormalVariates.py, plots of truncated normal variates are shown.    

## Source files used to generate examples
The source files used in each chapter are listed as follows.

**Chapter 1**

* MrfmBlurExample.py

**Chapter 3**

* EmgaussEmpiricalMapLazeReconstructorOnExample.py
* SimpleThresholdingReconstructorExample.py.
 
**Chapter 4**

* LarsReconstructorOnExample.py

**Chapter 5**

* MapPlazeGibbsSampleReconstructorOnExample.py



 