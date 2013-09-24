import math
import numpy as np
from Blur import AbstractBlur

class MrfmBlur(AbstractBlur):
    """
    STATIC METHODS
    """   
    @staticmethod
    def GetXyzMeshFor2d(xSpan, z0, numPoints):
        """
        Return a mesh for the 2d MRFM blur. The number of points in the x and y dimensions
        are equal and covers -xSpan to xSpan, while the z dimension is singleton with value z0.        
        """
        meshDelta = 2.0 * xSpan / (numPoints - 1.0)
        someEpsilon = 0.1
        xyzMesh = np.mgrid[-xSpan:(xSpan + someEpsilon * meshDelta):meshDelta,
                           -xSpan:(xSpan + someEpsilon * meshDelta):meshDelta,
                           z0:(z0 + someEpsilon):(.9 * someEpsilon)]
        return xyzMesh
    
    @staticmethod
    def GetXyzMeshFor3d(xSpan, z0, zMax, numPoints):        
        """
        Return a mesh for the 3d MRFM blur. The x and y dimensions cover [-xSpan, xSpan], while 
        the z dimension covers from z0 to zMax. The tuple numPoints is of length 3.                
        """        
        if len(numPoints) != 3:
            raise ValueError('numPoints is expected to have length 3')
        meshDelta = np.array((2.0 * xSpan, 2.0 * xSpan, zMax - z0)) / (np.array(numPoints) - 1.0)
        someEpsilon = 0.1
        xyzMesh = np.mgrid[-xSpan:(xSpan + someEpsilon * meshDelta[0]):meshDelta[0],
                           -xSpan:(xSpan + someEpsilon * meshDelta[1]):meshDelta[1],
                           z0:(zMax + someEpsilon * meshDelta[2]):meshDelta[2]]
        return xyzMesh
             
    @staticmethod
    def GetBmag(m, Bext, xMesh, yMesh, zMesh):
        xMeshPower2 = xMesh ** 2
        yMeshPower2 = yMesh ** 2
        zMeshPower2 = zMesh ** 2
        r = np.sqrt(xMeshPower2 + yMeshPower2 + zMeshPower2)
                
        rPower5 = r ** 5
        a1 = 3 * xMesh * zMesh * m / rPower5 
        a2 = 3 * yMesh * zMesh * m / rPower5
        a3 = (2 * zMeshPower2 - xMeshPower2 - yMeshPower2) * m / rPower5 + Bext
        
        Bmag = np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
        Bmag[np.where(r == 0)] = np.NaN
        return Bmag
    
    @staticmethod
    def GetG(m, Bext, xMesh, yMesh, zMesh):
        xMeshPower2 = xMesh ** 2
        yMeshPower2 = yMesh ** 2
        zMeshPower2 = zMesh ** 2
        r = np.sqrt(xMeshPower2 + yMeshPower2 + zMeshPower2)        
        
        rPower5 = r ** 5
        rPower7 = r ** 7
        rPower10 = rPower5 * rPower5
        rPower12 = rPower5 * rPower7

        n1 = -xMeshPower2 - yMeshPower2 + 2 * zMeshPower2
        mPower2 = m ** 2
        
        a1 = -90 * mPower2 * (xMesh * xMeshPower2) * zMeshPower2 / rPower12 - \
              90 * mPower2 * xMesh * yMeshPower2 * zMeshPower2 / rPower12 + \
              18 * mPower2 * xMesh * zMeshPower2 / rPower10
        a2 = -2 * m * xMesh / rPower5 - 5 * m * xMesh * n1 / rPower7
        a3 = Bext + m * n1 / rPower5
        a4 = 9 * mPower2 * xMeshPower2 * zMeshPower2 / rPower10 + 9 * mPower2 * yMeshPower2 * zMeshPower2 / rPower10
        a5 = a3 ** 2;

        numer = a1 + 2 * a2 * a3
        denom = 2 * np.sqrt(a4 + a5)
        G = numer / denom
        G[np.where(r == 0)] = np.NAN
        return G        
    
    @staticmethod
    def GetS(Bmag, Bres, gMesh):
        # S <- (Bres - BMag)/gMesh, but because Bmag can contains NaNs, while gMesh can contains zeros and/or NaNs, 
        # proceed in several steps            
        S = Bres - Bmag 

        # Carry out the division when G is finite and non-zero
        indGDefined = np.where(np.logical_and(np.logical_not(np.isnan(gMesh)), (gMesh != 0)))
        S[indGDefined] = S[indGDefined] / gMesh[indGDefined]        
        
        # Where G is NaN or zero, set S to NaN
        indGUndefined = np.where(np.logical_or(np.isnan(gMesh), (gMesh == 0)))
        S[indGUndefined] = np.NaN    
        
        return S        
    
    @staticmethod
    def GetPsfVerticalCantilever(m, Bext, Bres, xPk, xMesh, yMesh, zMesh):
        BmagMesh = MrfmBlur.GetBmag(m, Bext, xMesh, yMesh, zMesh)
        GMesh = MrfmBlur.GetG(m, Bext, xMesh, yMesh, zMesh)           
        sMesh = MrfmBlur.GetS(BmagMesh, Bres, GMesh)
        P = np.zeros(sMesh.shape, dtype=float)
        indSDefined = np.where(np.logical_not(np.isnan(sMesh)))
        P[indSDefined] = 1 - (sMesh[indSDefined] ** 2) / (xPk ** 2)        
        P[np.where(P < 0)] = 0
        return P * (GMesh ** 2)
            
    """
    CONSTANTS
    """
    INPUT_KEY_BEXT = 'Bext'
    INPUT_KEY_BRES = 'Bres'
    INPUT_KEY_SMALL_M = 'm'
    INPUT_KEY_XPK = 'xPk'
    INPUT_KEY_EPS = 'eps'
    INPUT_KEY_XMESH = 'xMesh'
    INPUT_KEY_YMESH = 'yMesh'
    INPUT_KEY_ZMESH = 'zMesh'

    BIG_M_DEFAULT = 1700        # Units is ??? Previously was set to 800.
    R0_DEFAULT = 60.17          # [nm]

    # Default values
    SMALL_M_DEFAULT = 4 * math.pi / 3 * BIG_M_DEFAULT * (R0_DEFAULT ** 3)   # [emu] assume spherical tip
    BEXT_DEFAULT = 2e4          # [Gauss]
    BRES_DEFAULT = 2.25e4       # [Gauss]
    XPK_DEFAULT = 0.2           # [nm] 
    
    # 2-d blur    
    BLUR_2D = 1    
    BLUR_3D = 2
                        
    def __init__(self, blurType, blurParametersDict):
        super(MrfmBlur, self).__init__()
        
        self._blurType = blurType        
        
        # Mesh parameters
        self._xMesh = blurParametersDict.get(MrfmBlur.INPUT_KEY_XMESH)
        self._yMesh = blurParametersDict.get(MrfmBlur.INPUT_KEY_YMESH)
        self._zMesh = blurParametersDict.get(MrfmBlur.INPUT_KEY_ZMESH)
                
        # Experimental conditions
        self.m = blurParametersDict.get(MrfmBlur.INPUT_KEY_SMALL_M, MrfmBlur.SMALL_M_DEFAULT)
        self.Bext = blurParametersDict.get(MrfmBlur.INPUT_KEY_BEXT, MrfmBlur.BEXT_DEFAULT)
        self.Bres = blurParametersDict.get(MrfmBlur.INPUT_KEY_BRES, MrfmBlur.BRES_DEFAULT)
        self.xPk = blurParametersDict.get(MrfmBlur.INPUT_KEY_XPK, MrfmBlur.XPK_DEFAULT)
        self._eps =  blurParametersDict.get(MrfmBlur.INPUT_KEY_EPS, 2.22e-16)
            
        # Initialize
        self._blurPsf = None
        self._blurShift = None
        self._psfSupport = None
        
        if ((blurType == MrfmBlur.BLUR_2D) or (blurType == MrfmBlur.BLUR_3D)):
            if ((self.X is not None) and (self.Y is not None) and (self.Z is not None)):
                # If the mesh is defined when the object is initialized, get the blur psf right away
                self._GetBlurPsf()           
        else:
            raise NotImplementedError("MrfmBlur type " + self._blurType + " hasn't been implemented")        
        
    """
    X, Y, Z
    """
    @property
    def X(self):
        return self._xMesh
        
    @property
    def Y(self):
        return self._yMesh
        
    @property
    def Z(self):
        return self._zMesh
        
    @property
    def PsfSupport(self):
        return self._psfSupport
            
    def _GetBlurPsf(self):
        if ((self.X is None) or (self.Y is None) or (self.Z is None)):
            raise UnboundLocalError('Cannot get psf since mesh definition is undefined')
        
        psf = MrfmBlur.GetPsfVerticalCantilever(self.m, self.Bext, self.Bres, self.xPk, self.X, self.Y, self.Z)
         
        if (self._blurType == MrfmBlur.BLUR_2D):
            psf = psf[:, :, 0]; # Reduce 3d to 2d array                          
            # Find support of psf in the x and y plane             
            psfSupport = np.where(psf > 2*self._eps)            
            # The blur shift is half of the max support in both x and y
            blurShift = [0, 0]; # Init
        elif (self._blurType == MrfmBlur.BLUR_3D):
            psfSupport = np.where(psf > 2*self._eps)
            blurShift = [0, 0, 0]
        else:
            raise NotImplementedError("MrfmBlur type " + self._blurType + " hasn't been implemented")
        
        self._blurPsf = psf / np.max(psf.flat)
        self._psfSupport = psfSupport
        for dimInd in range(len(blurShift)):
            blurShift[dimInd] = min(psfSupport[dimInd]) + math.floor((max(psfSupport[dimInd]) - min(psfSupport[dimInd])) / 2)                                   
        self._blurShift = tuple(blurShift)
        
    def BlurImage(self, theta):               
        self._thetaShape = theta.shape
        
        if (self._blurPsf is None):
            self._GetBlurPsf()
         
        if (len(self._thetaShape) != len(self._blurPsf.shape)):
            raise ValueError("Mismatch in theta's shape vs. the psf's shape")
        
        if (not np.all(self._thetaShape >= self._blurPsf.shape)):
            raise ValueError("theta's shape: " + str(self._thetaShape) + 
                             " must be at least as big as the blur's shape: " + str(self._blurPsf.shape)
                             ) 
                                
        if (self._blurType == MrfmBlur.BLUR_2D):                                                                   
            y = np.fft.ifft2(np.multiply(np.fft.fft2(self.BlurPsfInThetaFrame), np.fft.fft2(theta)))            
        elif (self._blurType == MrfmBlur.BLUR_3D):
            y = np.fft.ifftn(np.multiply(np.fft.fftn(self.BlurPsfInThetaFrame), np.fft.fftn(theta)))
        # Since _blurType was checked in __init__, there's no need to have an else to handle unimplemented
        # values.
        
        return y.real; # Discard the complex part
