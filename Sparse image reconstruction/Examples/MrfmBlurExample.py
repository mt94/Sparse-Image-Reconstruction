import pylab as plt
from AbstractExample import AbstractExample
from Sim.BmagLevelSurface import BmagLevelSurface
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer

class MrfmBlurExample(AbstractExample):
    def __init__(self, opti, numPointsInMesh, blurType, blurDesc=None):
        super(MrfmBlurExample,self).__init__('MrfmBlur example: ' + str(blurDesc))
        self._opti = opti
        assert (len(numPointsInMesh) == 1) or (len(numPointsInMesh) == 2)
        self._numPointsInMesh = numPointsInMesh        
        self._blurType = blurType
        self._mrfmBlurObj = None
    
    @property
    def Blur(self):
        return self._mrfmBlurObj
    
    def Plot(self, fignum):
        mrfmBlurObj = self._mrfmBlurObj
        if mrfmBlurObj is None:
            return; # Nothing to do
        
        if (self._blurType == MrfmBlur.BLUR_2D): 
            plt.figure(fignum)
            plt.imshow(mrfmBlurObj.BlurPsf, interpolation='none')        
            plt.xlabel('x'), plt.ylabel('y'), plt.colorbar() 
            plt.title(self.exampleDesc)
        else:
            # 3-d blur
            blurPsfShape = mrfmBlurObj.BlurPsf.shape
            # Plot the 3-d psf by plotting all x-y slices
            for zInd in range(blurPsfShape[2]):
                plt.figure(fignum + zInd)
                plt.imshow(mrfmBlurObj.BlurPsf[:, :, zInd], interpolation='none')        
                plt.xlabel('x'), plt.ylabel('y')
#                plt.colorbar()                
                plt.title(self.exampleDesc + ": slice " + str(zInd + 1))     
                        
        print(self.exampleDesc + ": psf has size: " + str(mrfmBlurObj.BlurPsf.shape))
                    
    """ Abstract method override """                
    def RunExample(self):     
        # Generate the mesh 
        if (self._blurType == MrfmBlur.BLUR_2D):       
            assert len(self._numPointsInMesh) == 1 
            xyzMesh = MrfmBlur.GetXyzMeshFor2d(self._opti.xSpan, 
                                               self._opti.z0, 
                                               self._numPointsInMesh[0]
                                               ) 
        elif (self._blurType == MrfmBlur.BLUR_3D):
            assert len(self._numPointsInMesh) == 2; # The #pts in the x and y dimensions are the same
            bmLevelSurface = BmagLevelSurface(self._opti.Bext, self._opti.Bres ** 2.0, self._opti.m)
            zMaxInMesh = bmLevelSurface.GetSupport['zSupport'][1] * 1.1
            xyzMesh = MrfmBlur.GetXyzMeshFor3d(self._opti.xSpan, 
                                               self._opti.z0, 
                                               zMaxInMesh,
                                               (self._numPointsInMesh[0], self._numPointsInMesh[0], self._numPointsInMesh[1])
                                               )
        else:
            raise NotImplementedError()
        
        blurParametersDict = {
                              MrfmBlur.INPUT_KEY_BEXT: self._opti.Bext,
                              MrfmBlur.INPUT_KEY_BRES: self._opti.Bres,
                              MrfmBlur.INPUT_KEY_SMALL_M: self._opti.m,
                              MrfmBlur.INPUT_KEY_XPK: self._opti.xPk,
                              MrfmBlur.INPUT_KEY_XMESH: xyzMesh[1],
                              MrfmBlur.INPUT_KEY_YMESH: xyzMesh[0],
                              MrfmBlur.INPUT_KEY_ZMESH: xyzMesh[2]
                              }
        
        # Create the MRFM blur      
        mb = MrfmBlur(self._blurType, blurParametersDict)        
        mb.GetBlurPsf()
        self._mrfmBlurObj = mb
        return self                    
        
if __name__ == "__main__":
    # Generate 2-d MRFM psf used in 04/s/psf_sim_sing.m
    example1Desc = '2d MRFM psf used in psf_sim_sing.m'
    opti = MrfmBlurParameterOptimizer()
    opti.CalcOptimalValues(3e4, 3)       
    #MrfmBlurExample(opti, (100,), MrfmBlur.BLUR_2D, example1Desc).RunExample().Plot(1)
    
    # Generate 2-d MRFM psf used in 04/f/sp_img_recon.m (less realistic parameters than those used in psf_sim_sing.m)
    example2Desc = '2d MRFM psf used in sp_img_recon.m'
    opti = MrfmBlurParameterOptimizer(deltaB0=100) 
    opti.bUseSmallerDeltaB0 = False;    # Use deltaB0 <- 100
    opti.CalcOptimalValues(1e4, 6, R0=4)
    #MrfmBlurExample(opti, (32,), MrfmBlur.BLUR_2D, example2Desc).RunExample().Plot(2)
            
    # Generate 3-d MRFM psf
    example3Desc = '3d MRFM psf used in pdb_sstart_psf.m'
    opti = MrfmBlurParameterOptimizer(deltaB0=100)
    opti.bUseSmallerDeltaB0 = False;    # Use delta B0 <- 100
    opti.CalcOptimalValues(1e4, 3)
    MrfmBlurExample(opti, (32, 14), MrfmBlur.BLUR_3D, example3Desc).RunExample().Plot(3)  
        
    # Call show to display the plots99
    plt.show()        
