import pylab as plt
from AbstractExample import AbstractExample
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer

class MrfmBlurExample(AbstractExample):
    def __init__(self, opti, numPointsInXyMesh, fignum):
        super(MrfmBlurExample,self).__init__('MrfmBlur example')
        self._opti = opti
        self._numPointsInXyMesh = numPointsInXyMesh
        self._fignum = fignum
        
    """ Abstract method override """                
    def RunExample(self):        
        # Use numpy.mgrid to generate 3-d grid mesh
        xyzMesh = MrfmBlur.GetXyzMeshFor2d(self._opti.xSpan, 
                                           self._opti.z0, 
                                           self._numPointsInXyMesh
                                           ) 
        # Create the MRFM blur      
        mb = MrfmBlur(MrfmBlur.BLUR_2D, 
                      {
                        MrfmBlur.INPUT_KEY_BEXT: self._opti.Bext,
                        MrfmBlur.INPUT_KEY_BRES: self._opti.Bres,
                        MrfmBlur.INPUT_KEY_SMALL_M: self._opti.m,
                        MrfmBlur.INPUT_KEY_XPK: self._opti.xPk,
                        MrfmBlur.INPUT_KEY_XMESH: xyzMesh[1],                                                 
                        MrfmBlur.INPUT_KEY_YMESH: xyzMesh[0],
                        MrfmBlur.INPUT_KEY_ZMESH: xyzMesh[2]
                       }               
                      )
        
        mb._GetBlurPsf()
        
        plt.figure(self._fignum)
        plt.imshow(mb.BlurPsf, interpolation='none')        
        plt.xlabel('x'), plt.ylabel('y')
        plt.colorbar()        
        print("Psf has size: " + str(mb.BlurPsf.shape))
        
if __name__ == "__main__":
    # Generate MRFM psf used in 04/s/psf_sim_sing.m
    opti = MrfmBlurParameterOptimizer()
    opti.CalcOptimalValues(3e4, 3)       
    ex = MrfmBlurExample(opti, 100, 1)
    ex.RunExample()
    plt.title('MRFM psf used in psf_sim_sing.m')
    
    # Generate MRFM psf used in 04/f/sp_img_recon.m (less realistic parameters than those used in psf_sim_sing.m)
    opti = MrfmBlurParameterOptimizer(deltaB0=100)
    opti.bUseSmallerR0 = True
    opti.bUseSmallerB0 = False
    opti.CalcOptimalValues(1e4, 6, R0=4)
    ex = MrfmBlurExample(opti, 32, 2)
    ex.RunExample()
    plt.title('MRFM psf used in sp_img_recon.m')
            
    plt.show()        