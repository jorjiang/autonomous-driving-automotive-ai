# autonomous-driving-automotive-ai
automotive-ai

Code for use in my autonomous driving research. Consists of largely-independent python packages
making heavy use of the numpy, scipy, and matplotlib libaries. 


## Packages used
* **hgmm**: 

    A package implementing Frank Havlak's hybrid gaussian mixture model algorithm.

    *Havlak, Frank, and Mark Campbell. "Discrete and continuous, probabilistic anticipation for autonomous robots in urban environments." IEEE Transactions on Robotics 30.2 (2014): 461-474.*

* **collision**:

    A package implementing Jason Hardy's fast collision probabilty algorithm between 
    two convex polygonal objects, one of which has an uncertain position.

    *Hardy, Jason, and Mark Campbell. "Contingency planning over probabilistic obstacle predictions for autonomous road vehicles." IEEE Transactions on Robotics 29.4 (2013): 913-929.*

* **myplot**:

    Helper package for plotting gaussians as heatmaps or covariance ellipses
