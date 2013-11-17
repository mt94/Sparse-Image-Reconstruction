from datetime import datetime
import logging
import math
import mpmath
import numpy as np
import pymc
import scipy.special as spspecial
import sys

from AbstractMcmcSampler import AbstractMcmcSampler
from McmcConstants import McmcConstants
from McmcIterationEvaluator import McmcIterationEvaluator
from Recon.AbstractReconstructor import AbstractReconstructor
from Systems.NumericalHelper import NumericalHelper

class PlazeGibbsSamplerReconstructor(AbstractMcmcSampler, AbstractReconstructor):
    """ 
    This class does not define an implementation for the abstract Estimate inherited from AbstractReconstructor.
    It's up to a derived class to do so. What's furnished here is the Gibbs Sampler implementation assuming that
    the image is i.i.d. positive-LAZE. 
    """
     
    def __init__(self, optimSettingsDict):
        super(PlazeGibbsSamplerReconstructor, self).__init__()

        if not(McmcConstants.INPUT_KEY_EPS in optimSettingsDict):
            raise KeyError('Missing keys in optimSettingsDict')        
        self.Eps = optimSettingsDict[McmcConstants.INPUT_KEY_EPS]        

        if not(McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT in optimSettingsDict):
            raise KeyError('Missing keys in optimSettingsDict')        
        self.hyperparameterPriorDict = optimSettingsDict[McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT]
            
        # Use defaults of 2000 iterations and 300 burn-in iterations    
        self.Iterations = optimSettingsDict.get(McmcConstants.INPUT_KEY_NUM_ITERATIONS, 2000)            
        self.BurninSamples = optimSettingsDict.get(McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES, 300)                    
        self.ThinningPeriod = optimSettingsDict.get(McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD, 1) # By default, don't do any thinning
            
        self.iterObserver = optimSettingsDict.get(McmcConstants.INPUT_KEY_ITERATIONS_OBSERVER)            
        self.nVerbose = optimSettingsDict.get(McmcConstants.INPUT_KEY_NVERBOSE, 0)
                    
        datetimeNow = datetime.now()

        # If no verbose turned on, set the level to Warning instead of Info
        if (self.nVerbose == 0):
            logLevel = logging.WARNING
        else:
            logLevel = logging.INFO
        
        logging.basicConfig(filename='PlazeGibbsSamplerReconstructor-{0}-{1}-{2}.log'.format(datetimeNow.year,
                                                                                             datetimeNow.month,
                                                                                             datetimeNow.day),
                            filemode="w+",
                            format='%(asctime)s %(message)s',                            
                            level=logLevel)                    
        
    # Constants for the static method _C 
    CONST_SQRT_HALF_PI = math.sqrt(math.pi/2)
    CONST_SQRT_2 = math.sqrt(2)        
    
    # Sample xInd|w, a, sigma^2, y, x\xInd
    def DoSamplingSpecificXConditionedAll(self, w, a, ind, fitErrExcludingInd, varLast, bLogDebug=False):
        assert (a > 0) and (w > 0) and (varLast > 0)                        
        
        etaIndSquared = varLast / self._hNormSquared[ind]       
        assert(etaIndSquared > 0)     
        dotProduct = np.dot(fitErrExcludingInd, self._h[:, ind])
        muIndComponents = (dotProduct / self._hNormSquared[ind], -etaIndSquared / a)
        muInd = sum(muIndComponents)
        
        """
        When y is big, calculating uInd is a challenge since there are numerical issues. We're 
        trying to multiply a very small number (which equals 0 due to finite floating point
        representation) and a very large number, which is prone to returning 0.        
        """  
        y = -muInd / PlazeGibbsSamplerReconstructor.CONST_SQRT_2 / math.sqrt(etaIndSquared)
                        
        uInd = (w / a) * \
            mpmath.sqrt(etaIndSquared) * PlazeGibbsSamplerReconstructor.CONST_SQRT_HALF_PI * mpmath.erfc(y) * \
            mpmath.exp(y * y)
                                               
        uIndFloat = float(uInd) # Convert to an ordinary Python float
        assert (uIndFloat > 0)
        
        if uIndFloat == float('inf'):
            wInd = 1
        else:
            wInd = uIndFloat / (uIndFloat + (1 - w))            
                                                                                                                
        if ((wInd < 0) or (wInd > 1)):
            raise ValueError('uInd is {0} and wInd is {1}'.format(uInd, wInd))                                    
        
        if pymc.rbernoulli(wInd):
            # With probability wInd, generate a sample from a truncated Gaussian r.v. (support (0,Inf))
#             xSample = pymc.rtruncated_normal(muInd, 1/etaIndSquared, a=0)[0]
            try:            
                xSample = NumericalHelper.RandomNonnegativeNormal(muInd, etaIndSquared)
            except:
                fmtString = "Caught exception at {0}: NNN({1}, {2}). Intm. calc.: {3}, {4}, {5}. Exception: {6}"
                msg = fmtString.format(ind,
                                       muInd, etaIndSquared,                                                
                                       muIndComponents[0],
                                       muIndComponents[1],
                                       varLast,
                                       sys.exc_info()[0])
                logging.error(msg)
                xSample = 0; # XXX: Due to a numerical problem
            else:
                # Check the value of xSample 
                if (xSample < 0):
                    fmtString = "Invalid xSample at {0}: NNN({1}, {2}) ~> sample {3}. Intm. calc.: {4}, {5}, {6}"
                    logging.error(fmtString.format(ind,
                                                   muInd, etaIndSquared, xSample,                                               
                                                   muIndComponents[0],
                                                   muIndComponents[1],
                                                   varLast))
                    # Don't throw an exception
    #                raise ValueError('xSample cannot be negative')
                    xSample = 0; # XXX: Also due to a numerical problem, but no exception raised   
        else:
            # With probability (1-wInd), generate 0
            xSample = 0
        
        if bLogDebug:
            fmtString = '      {0}/{1}: {2:.5e}, {3:.5f}={4:.5f}-{5:.5f}, {6:.5e}, {7:.5e}: {8:.5e}'
            logging.debug(fmtString.format(self._samplerIter,
                                           ind, 
                                           etaIndSquared, 
                                           muInd, 
                                           muIndComponents[0],
                                           -muIndComponents[1],
                                           uIndFloat, 
                                           wInd, 
                                           xSample))
            
        return xSample
                             
    # Sample x|w, a, sigma^2, y    
    def DoSamplingXConditionedAll(self, y, w, a, varLast, xLast, hxLast, bLogDebug=False):                
        M = xLast.size
        xNext = np.copy(xLast)
        hxNext = np.copy(hxLast) 

        #/// BEGIN SANITY
        discrepencyNormTol = self.Eps * 1e4
        
        hxLast2 = np.zeros(hxLast.shape)
        for ind in range(M):
            hInd = self._h[:, ind]            
            hxLast2[:, 0] += (xLast[ind] * hInd)

        discrepency = hxLast2 - hxLast            
        discrepencyNorm = math.sqrt(np.sum(discrepency * discrepency))
        hxLastNorm = math.sqrt(np.sum(hxLast * hxLast))
        hxLast2Norm = math.sqrt(np.sum(hxLast2 * hxLast2))            
                
        if (discrepencyNorm > discrepencyNormTol):
            raise RuntimeError('|dis| is {0} > tol={1}: |hxLast| = {2}, |hxLast2| = {3}'.format(discrepencyNorm,
                                                                                                discrepencyNormTol,
                                                                                                hxLastNorm, 
                                                                                                hxLast2Norm)
                               )
        #/// END SANITY
        
        # Sample each xInd                             
        for ind in range(M):
            hInd = self._h[:, ind]
#             hxExcludingInd = hxNext[:, 0] - xNext[ind] * hInd  # Temp var only needed for each iteration
            fitErrExcludingInd = y - (hxNext[:, 0] - hInd * xNext[ind])
            xIndNew = self.DoSamplingSpecificXConditionedAll(w, a, ind, fitErrExcludingInd, varLast, bLogDebug)                        
            hxNext[:, 0] = hxNext[:, 0] + hInd * (xIndNew - xNext[ind])
            xNext[ind] = xIndNew; # Replace the ind-th component with the newly sampled xInd            

        return (xNext, hxNext)

    # One iteration of the Gibbs sampler. Assume that _InitializeForSamplingX has been called.   
    def DoSamplingIteration(self, y):        
        # Setup
        try:
            xLast = self.xSeq[-1]
            varLast = self.varianceSeq[-1]
        except:
            raise NameError('Cannot access xLast and/or varLast')
        
        # n0 = #{ i : xLast[i] == 0} whereas n1 = ||xLast||_0        
        n0, n1 = NumericalHelper.CalculateNumZerosNonzeros(xLast, self.Eps)
        
        # Sample to get w|xLast and a|xLast,alpha
        wSample = pymc.rbeta(1 + n1, 1 + n0) 
        assert (wSample >= 0) and (wSample <= 1)               
        logging.info("  Samp. Iter. {0}, generating wSample ~ Beta({1},{2}) ... {3:.5f}".format(self._samplerIter, 1 + n1, 1 + n0, wSample))
        
        igShapeForA = n1 + self.hyperparameterPriorDict['alpha0']
        xLastL1Norm = np.sum(np.abs(xLast))
        igScaleForA = xLastL1Norm + self.hyperparameterPriorDict['alpha1']
        
        assert (igShapeForA > 0) and (igScaleForA > 0)
       
        bSampleGenerated = False
        aSample = None
        
        for tryInd in range(5, 0, -1):     
            try:
                aSample = pymc.rinverse_gamma(igShapeForA, igScaleForA)
                bSampleGenerated = True                
            except (ZeroDivisionError, OverflowError) as e:
                logging.error("Couldn't generate aSample ~ IG({0},{1}) using n0={2}, n1={3}: {4}".format(igShapeForA, igScaleForA, n0, n1, e.message))
                # Only raise the exception if repeated attempts failed
                if tryInd == 0:
                    raise            
            if bSampleGenerated is True:
                break
                    
        logging.info("  Samp. Iter. {0}, generating aSample ~ IG({1:.4f},{2:.4f}) ... {3:.4e}".format(self._samplerIter, igShapeForA, igScaleForA, aSample))
                    
        self.hyperparameterSeq.append({'w' : wSample, 'a' : aSample})        
        
        # Sample to get x_i, 1 <= i <= M. The method DoSamplingXConditionedAll updates self.xSeq and self._mappedX        
        xNext, hxNext = self.DoSamplingXConditionedAll(y, wSample, aSample, varLast, xLast, self.hx, self.nVerbose > 0)
                
        # Sample to get variance
        yErr = y - hxNext[:, 0]
        igShapeForVariance = y.size / 2
        igScaleForVariance = np.sum(yErr * yErr) / 2
        varianceSample = pymc.rinverse_gamma(igShapeForVariance, igScaleForVariance)        
        logging.info("  Samp. Iter. {0}, generating varianceSample ~ IG({1:.4f},{2:.4f}) ... {3:.4e}".format(
                                                                                                             self._samplerIter,
                                                                                                             igShapeForVariance, 
                                                                                                             igScaleForVariance, 
                                                                                                             varianceSample
                                                                                                             ))        
        self.varianceSeq.append(varianceSample)        

        self.iterObserver.UpdateState({
                                       McmcIterationEvaluator.STATE_KEY_COUNT_ITER: self._samplerIter,
                                       McmcIterationEvaluator.STATE_KEY_X_ITER: xNext,
                                       McmcIterationEvaluator.STATE_KEY_HX_ITER: hxNext,
                                       McmcIterationEvaluator.STATE_KEY_W_ITER: wSample,
                                       McmcIterationEvaluator.STATE_KEY_A_ITER: aSample,
                                       McmcIterationEvaluator.STATE_KEY_NOISEVAR_ITER: varianceSample
                                       })
        
        self.xSeq.append(xNext)
        assert len(self.xSeq) == len(self.varianceSeq)
        assert len(self.xSeq) == (len(self.hyperparameterSeq) + 1) 
        
        self.hx = hxNext

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
        FLaplaceDistInv = lambda x: (-a * np.sign(x - 0.5) * np.log(1 - 2 * np.abs(x - 0.5)))
        plazeSample = FLaplaceDistInv(0.5 + 0.5 * np.random.uniform(size=(xLen, 1))) 
        xSample = bIsZero * plazeSample
        return xSample
    
    # Sample var|varMin, varMax
    @staticmethod
    def DoSamplingPriorVariancePrior(varMin, varMax):
        """ 
        In reality, the variance prior is improper since f(sigma^2) \propto 1/sigma^2. For 
        practical reasons, impose a min/max value.
        """
        assert (varMin > 0) and (varMin < varMax)
        return varMin*(varMax/varMin)**np.random.uniform() 
    
    @staticmethod
    def ComputePosteriorProb(y, convMatrixObj, theta, aPriorDict):
        assert ('alpha0' in aPriorDict) and ('alpha1' in aPriorDict)
        thetaFlat = theta.flat        
        if not np.all(thetaFlat > 0):
            return 0
        err = y - convMatrixObj.Multiply(theta)
        n0, n1 = PlazeGibbsSamplerReconstructor.CalculateNumZeros(thetaFlat)
        return spspecial.beta(1 + n1, 1 + n0) / \
               np.sum(err * err) * \
               spspecial.gamma(n1 + aPriorDict['alpha0']) / \
               (np.sum(np.abs(thetaFlat)) + aPriorDict['alpha1']) ** (n1 + aPriorDict['alpha0'])                  
                
    """
    ## 
    ## Implementation of abstract methods from AbstractMcmcSampler
    ## 
    """
                
    def SamplerSetup(self, convMatrixObj, initializationDict):
        """ This method must be called before SamplerRun """                
        if not('init_theta' in initializationDict) or not('init_var' in initializationDict):
            raise KeyError('Initialization dictionary missing keys init_theta and/or init_var')
        
        self.xSeq = [] # Contains x^{(t)}, t=0, ..., T
        x0 = np.array(initializationDict['init_theta']) # Expect x0 to be either a 2-d or 3-d array. 
                                                        # Notice nomenclature x here instead of theta.
                                                        # Nonetheless, use theta in the key.
        M = x0.size       
        self.xSeq.append(np.reshape(x0, (M, 1))) # Reshape x0 into a column array
        
        self.varianceSeq = [] # Contains \sigma^2^{(t)}, t=0, ..., T
        assert initializationDict['init_var'] >= 0
        self.varianceSeq.append(initializationDict['init_var'])
        
        self.hyperparameterSeq = [] # Contains hyperparameter estimates, t=1, ..., T                    
        
        # This will return a 2-d or 3-d matrix, so it'll have to be reshaped into a vector
        forwardMap = lambda x: convMatrixObj.Multiply(np.reshape(x, x0.shape))        
        self._h = np.zeros((M, M))
        self._hNormSquared = np.zeros((M,))
        for ind in range(M):
            eInd = np.zeros((M,1))         
            eInd[ind] = 1   
            tmp = forwardMap(eInd)            
            self._h[:, ind] = np.reshape(tmp, (M,)) 
            self._hNormSquared[ind] = np.dot(self._h[:,ind], self._h[:,ind])
            assert self._hNormSquared[ind] > 0
            
        self.hx = np.reshape(forwardMap(x0), (M, 1))
        
        self.bSamplerRun = False        
        
    def SamplerRun(self, y):
        """ This method is called after SamplerSetup in order to generates the samples of the MC """        
        if not(hasattr(self, 'bSamplerRun')) or (self.bSamplerRun):
            raise StandardError("Doesn't seem like SamplerSetup was called") 

        self._samplerIter = 0
        yFlat = np.array(y.flat)
        
        for iterNum in range(self.Iterations + self.BurninSamples):
            self._samplerIter = iterNum + 1
            self.DoSamplingIteration(yFlat)
            
        self.bSamplerRun = True
            
    def SamplerGet(self, elementDesc, maxNumSamples = float('inf')):
        """ 
        Take into account self.{BurninSamples, ThinningPeriod} when returning xSeq
        """ 
        if not(hasattr(self, 'bSamplerRun')) or (not self.bSamplerRun):
            raise StandardError("Doesn't seem like SamplerRun was called")
        else:
            # Run checks to be sure that xSeq, varianceSeq, hyperparameterSeq are of the expected lengths
            assert len(self.xSeq) == (self.Iterations + self.BurninSamples + 1)
            assert len(self.varianceSeq) == len(self.xSeq)
            assert len(self.hyperparameterSeq) == (len(self.xSeq) - 1)
                        
        # If elementDesc isn't supported, return an empty list
        samples = {'theta': self.xSeq[(self.BurninSamples + 1)::self.ThinningPeriod],
                   'variance': self.varianceSeq[(self.BurninSamples + 1)::self.ThinningPeriod],
                   'hyperparameter': self.hyperparameterSeq[self.BurninSamples::self.ThinningPeriod]
                   }.get(elementDesc.lower(), [])
                   
        # Cap the number of returned samples?
        if len(samples) > maxNumSamples:
            return samples[:int(maxNumSamples)]
        else:
            return samples
            
                
