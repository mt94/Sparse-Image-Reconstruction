import math
from MrfmBlur import MrfmBlur

class MrfmBlurParameterOptimizer(object):
        
    """
    Constants
    """
    OPTIMAL_R0_FACTOR = 3
    OPTIMAL_X_FACTOR = -( math.sqrt(35) - math.sqrt(19) )/4 # optimal x-pos factor for max G
    OPTIMAL_G_FACTOR = 2.7428                               # max factor for G in terms of m, z0
    
    DELTA_B0 = 50           # [Gauss]
    DELTA_B0_SMALLER = 20   # [Gauss] Is unrealistic, however facilitates computation
    
    XSPAN_DEFAULT = 10      # [nm]
    Z0_DEFAULT = 83.5       # [nm]

    _A1 = -(2 - OPTIMAL_X_FACTOR ** 2) / (1 + OPTIMAL_X_FACTOR ** 2) ** 2.5
    _A2 = -9 * OPTIMAL_X_FACTOR ** 2 / (1 + OPTIMAL_X_FACTOR ** 2) ** 5;

    def __init__(self, mrfmBlur=None):
        super(MrfmBlurParameterOptimizer,self).__init__()
        
        # Initialize with values from the MrfmBlur object. If mrfmBlur isn't specified, 
        # take the default values from the MrfmBlur class
        if (mrfmBlur is not None): 
            self.m = mrfmBlur.m
            self.Bext = mrfmBlur.Bext
            self.Bres = mrfmBlur.Bres
            self.xPk = mrfmBlur.xPk
        else:
            self.m = MrfmBlur.SMALL_M_DEFAULT
            self.Bext = MrfmBlur.BEXT_DEFAULT
            self.Bres = MrfmBlur.BRES_DEFAULT
            self.xPk = MrfmBlur.XPK_DEFAULT
            
        # Defaults
        self.M = MrfmBlur.BIG_M_DEFAULT
        self.R0 = MrfmBlur.R0_DEFAULT
        self.xSpan = MrfmBlurParameterOptimizer.XSPAN_DEFAULT
        self.z0 = MrfmBlurParameterOptimizer.Z0_DEFAULT
        self.d = None
        self.xOpt = None
        self.GMax = None
      
        # Used for CalcOptimalValues  
        self.bUseSmallerR0 = True
        self.bUseSmallerB0 = True  
                          
    def CalcOptimalValues(self, Bres, d):
        
        self.d = d
        self.Bres = Bres
        
        if (self.bUseSmallerR0):
            self.R0 = d
        else:
            self.R0 = self.OPTIMAL_R0_FACTOR * d
            
        self.m = 4 * math.pi / 3 * self.M * (self.R0 ** 3)
        self.z0 = self.R0 + d
        
        # Compute xPk
        self.xOpt = self.OPTIMAL_X_FACTOR*self.z0                
        self.GMax = self.OPTIMAL_G_FACTOR * self.m / (self.z0 ** 4)
        
        if (self.bUseSmallerB0):
            deltaB0 = self.DELTA_B0_SMALLER
        else:
            deltaB0 = self.DELTA_B0
        
        self.xSpan = math.fabs( 2*self.xOpt )    
        self.xPk = deltaB0 / self.GMax
        
        # Compute Bext
        aa = self._A1 * (self.m / self.z0 ** 3)
        bb = math.sqrt(Bres ** 2 + self._A2 * (self.m / self.z0 ** 3) ** 2)
        quadSol1 = aa + bb
        quadSol2 = aa - bb        
        
        # Check that there is exactly one strictly positive and one strictly negative solution
        assert ((quadSol1 > 0) and (quadSol2 < 0)) or ((quadSol1 < 0) and (quadSol2 > 0))
         
        if (quadSol1 > 0):
            self.Bext = quadSol1
        elif (quadSol2 > 0):
            self.Bext = quadSol2
        
            
        
