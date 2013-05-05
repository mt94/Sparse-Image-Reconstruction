import math
import numpy as np
from Blur import AbstractBlur

class MrfmBlur(AbstractBlur):
    """
    STATIC METHODS
    """
    @staticmethod
    def GetXyzMesh(xRange, yRange, zRange):
        AllMeshes = np.mgrid(xRange, yRange, zRange)
        return AllMeshes[0], AllMeshes[1], AllMeshes[2]
        
    @staticmethod
    def GetBmag(m, Bext, xMesh, yMesh, zMesh):
        r = np.sqrt(xMesh ** 2 + yMesh ** 2 + zMesh ** 2)
        rPower5 = r ** 5
        a1 = 3 * xMesh * zMesh / rPower5 * m
        a2 = 3 * yMesh * zMesh / rPower5 * m
        a3 = (2 * zMesh ** 2 - xMesh ** 2 - yMesh ** 2) / rPower5 * m + Bext
        return np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
    
    @staticmethod
    def GetG(m, Bext, xMesh, yMesh, zMesh):
        r = np.sqrt(xMesh ** 2 + yMesh ** 2 + zMesh ** 2)
        rPower12 = r ** 12
        rPower10 = r ** 10
        rPower7 = r ** 7
        rPower5 = r ** 5
        
        xMeshPower2 = xMesh ** 2
        yMeshPower2 = yMesh ** 2
        zMeshPower2 = zMesh ** 2
        n1 = -xMeshPower2 - yMeshPower2 + 2 * zMeshPower2
        mPower2 = m ** 2
        
        a1 = -90 * mPower2 * (xMesh * xMeshPower2) * zMeshPower2 / rPower12 - \
              90 * mPower2 * xMesh * yMeshPower2 * zMeshPower2 / rPower12 + \
              18 * mPower2 * xMesh * zMeshPower2 / rPower10
        a2 = -2 * m * xMesh / rPower5 - 5 * m * xMesh * n1 / rPower7
        a3 = Bext + m * n1 / rPower5
        a4 = 9 * mPower2 * xMeshPower2 * zMeshPower2 / rPower10 + 9 * mPower2 * yMeshPower2 * zMeshPower2 / rPower10
        a5 = a3 ** 2;

        return (a1 + 2 * a2 * a3) / (2 * np.sqrt(a4 + a5))
    
    @staticmethod
    def GetS(Bmag, Bres, gMesh):
        return (Bres - Bmag) / gMesh
    
    @staticmethod
    def GetPsfVerticalCantilever(m, Bext, Bres, xPk, xMesh, yMesh, zMesh):
        BmagMesh = MrfmBlur.GetBmag(m, Bext, xMesh, yMesh, zMesh)
        GMesh = MrfmBlur.GetG(m, Bext, xMesh, yMesh, zMesh)
        sMesh = MrfmBlur.GetS(BmagMesh, Bres, GMesh)
        tmpMesh = 1 - (sMesh ** 2) / (xPk ** 2)
        tmpMesh[np.where(tmpMesh < 0)] = 0
        return tmpMesh * (GMesh ** 2)
            
    """
    CONSTANTS
    """
    INPUT_KEY_BEXT = 'Bext'
    INPUT_KEY_BRES = 'Bres'
    INPUT_KEY_SMALL_M = 'm'
    INPUT_KEY_XPK = 'xPk'

    BIG_M_DEFAULT = 1700        # Units is ??? Previously was set to 800.
    R0_DEFAULT = 60.17          # [nm]

    # Default values
    SMALL_M_DEFAULT = 4 * math.pi / 3 * BIG_M_DEFAULT * (R0_DEFAULT ** 3)   # [emu] assume spherical tip
    BEXT_DEFAULT = 2e4          # [Gauss]
    BRES_DEFAULT = 2.25e4       # [Gauss]
    XPK_DEFAULT = 0.2           # [nm] 
    
    # 2-d blur    
    BLUR_2D = 1    
                        
    def __init__(self, blurType, blurParametersDict):
        super(MrfmBlur, self).__init__()
        self._blurType = blurType        
        self._xMesh = None
        self._yMesh = None
        self._zMesh = None  
        
        self.m = blurParametersDict.get(MrfmBlur.INPUT_KEY_SMALL_M, MrfmBlur.SMALL_M_DEFAULT)
        self.Bext = blurParametersDict.get(MrfmBlur.INPUT_KEY_BEXT, MrfmBlur.BEXT_DEFAULT)
        self.Bres = blurParametersDict.get(MrfmBlur.INPUT_KEY_BRES, MrfmBlur.BRES_DEFAULT)
        self.xPk = blurParametersDict.get(MrfmBlur.INPUT_KEY_XPK, MrfmBlur.XPK_DEFAULT) 
        
    """
    X, Y, Z
    """
    @property
    def X(self):
        return self._xMesh
    @X.setter
    def X(self, value):
        self._xMesh = value
        
    @property
    def Y(self):
        return self._yMesh
    @Y.setter
    def Y(self, value):
        self._yMesh = value
        
    @property
    def Z(self):
        return self._zMesh
    @Z.setter
    def Z(self, value):
        self._zMesh = value
        
    def BlurImage(self, theta):   

        
        if (self._blurType == MrfmBlur.BLUR_2D):
            self._thetaShape = theta.shape
            if len(self._thetaShape) != 2:
                raise ValueError('BLUR_2D requires theta to be 2-d')
            
            psf = MrfmBlur.GetPsfVerticalCantilever(self.m, self.Bext, self.Bres, self.xPk, self.X, self.Y, self.Z) 
            psf2d = psf[:, :, 0]                
            self._blurPsf = psf2d
            
            if (not np.all(self._thetaShape >= psf2d.shape)):
                raise ValueError("theta's shape must be at least as big as the blur's shape")            
                                   
            # Calculate the shift induced by convolving theta with the psf
            
            y = np.fft.ifft2(np.multiply(np.fft.fft2(self.BlurPsfInThetaFrame), np.fft.fft2(theta)))
            return y.real                        
        else:
            raise NotImplementedError("MrfmBlur type " + self._blurType + " hasn't been implemented")
