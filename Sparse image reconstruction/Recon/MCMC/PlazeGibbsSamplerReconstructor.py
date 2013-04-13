import math
import numpy as np
import pymc

from AbstractMcmcSampler import AbstractMcmcSampler
from McmcConstants import McmcConstants
from Recon.AbstractReconstructor import AbstractReconstructor

class PlazeGibbsSamplerReconstructor(AbstractMcmcSampler, AbstractReconstructor):
    """ 
    This class does not define an implementation for the abstract Estimate inherited from AbstractReconstructor.
    It's up to a derived class to do so. What's furnished here is the Gibbs Sampler implementation assuming that
    the image is i.i.d. positive-LAZE. 
    """
     
    def __init__(self, optimSettingsDict):
        super(PlazeGibbsSamplerReconstructor, self).__init__()
            
        # Use defaults of 2000 iterations and 300 burn-in iterations    
        self.Iterations = optimSettingsDict[McmcConstants.INPUT_KEY_NUM_ITERATIONS] \
            if McmcConstants.INPUT_KEY_NUM_ITERATIONS in optimSettingsDict \
            else 2000
        self.BurninSamples = optimSettingsDict[McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES] \
            if McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES in optimSettingsDict \
            else 300           
             
        # By default, don't do any thinning, although this isn't recommended
        self.ThinningPeriod = optimSettingsDict[McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD] \
            if McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD in optimSettingsDict \
            else 1
        
        if not(McmcConstants.INPUT_KEY_EPS in optimSettingsDict):
            raise KeyError('Missing keys in optimSettingsDict')
        else:
            self.Eps = optimSettingsDict[McmcConstants.INPUT_KEY_EPS]    
            
    # Constants for the static method _C 
    _C_1 = math.sqrt(math.pi/2)
    _C_2 = math.sqrt(2)
    
    @staticmethod
    def _C(m, sSquared):
        s = math.sqrt(sSquared)
        return PlazeGibbsSamplerReconstructor._C_1 * s * (1 + math.erf(m / PlazeGibbsSamplerReconstructor._C_2 / s))
    
            
        
    # Sample xInd|w, a, sigma^2, y, x\xInd
    def _DoSamplingSpecificXConditionedAll(self, w, a, ind, errInd):
        etaSquaredInd = self.varianceSeq[-1] / self._hNormSquared[ind]
        muInd = etaSquaredInd * (np.dot(errInd, self._h[:, ind]) / self.varianceSeq[-1] - 1 / a)
        uInd = w / a * PlazeGibbsSamplerReconstructor._C(muInd, etaSquaredInd) * math.exp(muInd * muInd / 2 / etaSquaredInd)
        wInd = uInd / (uInd + (1 - w))
        assert (wInd >= 0) and (wInd <= 1)
        if pymc.rbernoulli(wInd):
            # Generate a sample from a truncated Gaussian r.v. (support (0,Inf))
            return pymc.rtruncated_normal(muInd, 1/etaSquaredInd, 0, np.inf)
        else:
            return 0
                             
    # Sample x|w, a, sigma^2, y    
    def _DoSamplingXConditionedAll(self, y, w, a):
        try:
            xLast = self.xSeq[-1]
        except:
            raise NameError('Cannot access xLast')
                
        M = xLast.size
        xNext = xLast;
                
        # Initially, this contains the forward map applied to xLast
        vecT = self._mappedX 
                
        for ind in range(M):
            vecTInBetween = vecT - xLast[ind] * self._h[:, ind]
            errInd = y - vecTInBetween
            xInd = self._DoSamplingSpecificXConditionedAll(w, a, ind, errInd)
            xNext[ind] = xInd
            vecT = vecTInBetween + xInd * self._h[:, ind]            

        # Add to self.xSeq
        self.xSeq.append(xNext)
        
        # Update T(xNext)
        self._mappedX = vecT

    # One iteration of the Gibbs sampler. Assume that _InitializeForSamplingX has been called.   
    def _DoSamplingIteration(self, y):        
        # Setup
        try:
            xLast = self.xSeq[-1]
        except:
            raise NameError('Cannot access xLast')
        
        I0 = np.where(np.abs(xLast) < self.Eps)[0]
        n0 = I0.size
        ##I1 = np.setdiff1d(np.arange(xLast.size), I0)
        ##n1 = I1.size
        n1 = xLast.size - n0
        
        # Sample to get w|xLast and a|xLast,alpha
        wSample = pymc.rbeta(1 + n1, 1 + n0)        
        aSample = pymc.rinverse_gamma(n0 + self.HyperparameterPriorDict['alpha0'], n1 + self.HyperparameterPriorDict['alpha1'])
        self.hyperparameterSeq.append({'w' : wSample, 'a' : aSample})
        
        # Sample to get x_i, 1 <= i <= M. The method _DoSamplingXConditionedAll updates self.xSeq
        self._DoSamplingXConditionedAll(y, wSample, aSample)
        
        # Sample to get variance
        err = y - self._mappedX
        varianceSample = pymc.rinverse_gamma(y.size / 2, np.sum(err * err) / 2)
        self.varianceSeq.append(varianceSample)

    """ Public methods """
    
    # Sample x|w, a
    def DoSamplingXPrior(self, w, a, M=None):
        if M is not None:
            xLen = M
        else:
            try:
                xLast = self.xSeq[-1]
                xLen = xLast.size
            except:
                raise UnboundLocalError('Unable to determine length of x')
        assert (w >= 0) and (w <= 1)
        bIsZero = pymc.rbernoulli(w, (xLen, 1))
        FLaplaceDistInv = lambda x:-a * np.sign(x - 0.5) * np.log(1 - 2 * np.abs(x - 0.5))
        plazeSample = FLaplaceDistInv(0.5 + 0.5 * np.random((xLen, 1))) 
        xSample = bIsZero * plazeSample
        return xSample
    
    @staticmethod
    def ComputePosteriorProb(y, convMatrixObj, theta):   
         
                
    """ Implementation of abstract methods from AbstractMcmcSampler """
                
    def SamplerSetup(self, convMatrixObj, initializationDict):
        """ This method must be called before SamplerRun """                
        if not('init_x' in initializationDict) or not('init_var' in initializationDict):
            raise KeyError('Initialization dictionary missing keys')
        
        self.xSeq = [] # Contains x^{(t)}, t=0, ..., T
        x0 = np.array(initializationDict['init_x']) # Expect x0 to be either a 2-d or 3-d array
        M = x0.size       
        self.xSeq.append(np.reshape(x0, (M, 1))) # Reshape x0 into a column array
        
        self.varianceSeq = np.zeros((self.Iterations,)) # Contains \sigma^2^{(t)}, t=0, ..., T
        self.varianceSeq[0] = initializationDict['init_var']
        
        self.hyperparameterSeq = [] # Contains hyperparameter estimates, t=1, ..., T                    
        
        # This will return a 2-d or 3-d matrix, so it'll have to be reshaped into a vector
        forwardMap = lambda x: convMatrixObj.Multiply(np.reshape(x, x0.shape))
        assert len(self.xSeq) == 1
        xLast = self.xSeq[0]        
        M = xLast.size
        
        self._h = np.zeros((M,M))
        self._hNormSquared = np.zeros((M,))
        for ind in range(M):
            eInd = np.zeros((M,1))         
            eInd[ind] = 1   
            self._h[:, ind] = np.reshape(forwardMap(eInd), (M, 1)) 
            self._hNormSquared[ind] = np.sum(self._h[:,ind] * self._h[:,ind])
            
        self._mappedX = np.reshape(forwardMap(xLast), (M, 1))
        
        self.bSamplerRun = False        
        
    def SamplerRun(self, y):
        """ This method is called after SamplerSetup in order to generates the samples of the MC """        
        if not(hasattr(self, 'bSamplerRun')) or (self.bSamplerRun):
            raise StandardError("Doesn't seem like _SamplerSetup was called") 

        yFlat = np.array(y.flat)           
                                                             
        for dummy in range(self.Iterations + self.BurninSamples):
            self._DoSamplingIteration(yFlat)
            
        self.bSamplerRun = True
            
    def SamplerGet(self, elementDesc):
        """ 
        Take into account self.{BurninSamples, ThinningPeriod} when returning xSeq
        """ 
        if not(hasattr(self, 'bSamplerRun')) or (not self.bSamplerRun):
            raise StandardError("Doesn't seem like _SamplerRun was called")
        else:
            # Run checks to be sure that xSeq, varianceSeq, hyperparameterSeq are of the expected lengths
            assert len(self.xSeq) == (self.Iterations + self.BurninSamples + 1)
            assert len(self.varianceSeq) == len(self.xSeq)
            assert len(self.hyperparameterSeq) == (len(self.xSeq) - 1)
                        
        # If elementDesc isn't supported, return an empty list
        return {'x': self.xSeq[(self.BurninSamples + 2)::self.ThinningPeriod],
                'variance': self.varianceSeq[(self.BurninSamples + 2)::self.ThinningPeriod],
                'hyperparameter': self.hyperparameterSeq[(self.BurninSamples + 1)::self.ThinningPeriod]
                }.get(elementDesc.lower(), [])
         
        
        
        
        
        
        
        
        
        
        
    
    
