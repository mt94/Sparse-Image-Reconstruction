import math
import numpy as np

class BmagLevelSurface(object):
    """
    Constants
    """
    DEFAULT_NUM_POINTS_IN_THETA_SPACE = 90
    DEFAULT_NUM_POINTS_IN_PHI_SPACE = 45
    
    def __init__(self, Bext, BmagSquared, m, 
                 R0=None,
                 maxPhiValue=math.pi/2.0, 
                 numPointsInThetaSpace=DEFAULT_NUM_POINTS_IN_THETA_SPACE, 
                 numPointsInPhiSpace=DEFAULT_NUM_POINTS_IN_PHI_SPACE):
        """
        Bext, m -- parameters of the experiment
        BmagSquared -- Bmag^2 for which we want to plot the level surface
        """        
        super(BmagLevelSurface, self).__init__()
                
        assert BmagSquared >= math.pow(Bext, 2.0)
        self._Bext = Bext
        self._c = BmagSquared
        
        self._m = m
        self._R0 = R0
        
        if ((maxPhiValue < 0) or (maxPhiValue > math.pi/2.0)):
            raise ValueError('Invalid value of maxPhiValue: ' + str.format('{0:.4f}', maxPhiValue))
        else:
            self._maxPhiValue = maxPhiValue
        
        self._numPointsInThetaSpace = numPointsInThetaSpace
        self._numPointsInPhiSpace = numPointsInPhiSpace
        
        self.CreatePlot(2.0*math.pi, self._maxPhiValue)
        
        """
        If R0 is given, then we're only interested in points for which z > R0. There may be NO
        points that satisfy this condition.
        """
        if R0 is not None:
            if self.GetSupport['zSupport'][1] < R0:
                raise ValueError('R0 is larger than z values of level surface')
            # R0 is a valid input. What can be done now is to find the value of phi that 
            # corresponds to R0, and then call CreatePlot again with this as maxPhi. In
            # this way, one gets afiner level surface given the restriction  of z > R0.         
            raise NotImplementedError()
          
    def CreatePlot(self, maxTheta, maxPhi):
        theta, phi = np.meshgrid(np.linspace(0, maxTheta, self._numPointsInThetaSpace),
                                 np.linspace(0, maxPhi, self._numPointsInPhiSpace)
                                 )        
        m2 = self._m ** 2
        sinPhi = np.sin(phi)
        sinTheta = np.sin(theta)
        
        a = (self._Bext ** 2 - self._c)
        b = 2 * self._m * self._Bext * (2 - 3 * np.power(sinPhi, 2))        
        c = 9 * m2 * (1 - np.power(sinTheta, 4)) * np.power(sinPhi, 4) + \
            3 * m2 * (3 * np.power(sinTheta, 2) - 4) * np.power(sinPhi, 2) + \
            4 * m2
        
        D = b**2 - 4*a*c
        #quadSol1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        #quadSol1PowerOneThird = np.power(quadSol1, 1.0 / 3.0)
        quadSol1 = np.where(D >= 0, (-b + np.sqrt(D))/(2*a), None)        
        #quadSol2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        #quadSol2PowerOneThird = np.power(quadSol2, 1.0 / 3.0)
        quadSol2 = np.where(D >= 0, (-b - np.sqrt(D))/(2*a), None)
        
        radius = np.where(np.logical_and(quadSol1 > 0, quadSol2 < 0), quadSol1, 0) + \
            np.where(np.logical_and(quadSol1 < 0, quadSol2 > 0), quadSol2, 0)
        radius = np.power(radius, 1.0/3.0)
            
        self._plotData = (theta, phi, radius)
        
    """
    Properties that can be accessed once CreatePlot has been called.
    """
    
    @property
    def SurfaceInSphericalCoordinates(self):
        # Gets the level surface in polar coordinates
        return self._plotData
    
    @property
    def SurfaceInCartesianCoordinates(self):
        # Gets the level surface in Cartesian coordinates
        (theta, phi, radius) = self._plotData
        sinPhi = np.sin(phi)
        return (radius * sinPhi * np.cos(theta), radius * sinPhi * np.sin(theta), radius * np.cos(phi))
    
    @property
    def GetSupport(self):
        # Returns the x, y, z support of the level surface
        (x, y, z) = self.SurfaceInCartesianCoordinates
        return { 'xSupport': (np.min(x), np.max(x)),
                 'ySupport': (np.min(y), np.max(y)),
                 'zSupport': (np.min(z), np.max(z))
                }
        


            
        
    
