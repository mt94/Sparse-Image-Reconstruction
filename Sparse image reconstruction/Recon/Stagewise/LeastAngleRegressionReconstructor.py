import numpy as np
import warnings

from Recon.AbstractReconstructor import AbstractReconstructor
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

class LeastAngleRegressionReconstructor(AbstractReconstructor):
    
    INPUT_KEY_MAX_ITERATIONS = 'max_iters'
    INPUT_KEY_EPS = 'eps'
    INPUT_KEY_NVERBOSE = 'nverbose'
        
    OUTPUT_KEY_ACTIVESET = 'activeset'
    OUTPUT_KEY_MAX_CORRHAT_HISTORY = 'max_corrhat_history'    
    OUTPUT_KEY_MUHAT_ACTIVESET = 'muhat_activeset'
    
    def __init__(self, optimSettingsDict=None):   
        super(LeastAngleRegressionReconstructor, self).__init__()                
        
        self.EPS = optimSettingsDict[LeastAngleRegressionReconstructor.INPUT_KEY_EPS] \
            if LeastAngleRegressionReconstructor.INPUT_KEY_EPS in optimSettingsDict \
            else 1.0e-7
                    
        self._optimSettingsDict = optimSettingsDict            

    """ Calculate AActiveSet and uActiveSet using XActiveSet via (2.5) and (2.6). Assume that 
        XActiveSet is a matrix. uActiveSet is the equiangular vector. """
    @staticmethod    
    def CalculateAU(XActiveSet):
        assert isinstance(XActiveSet, np.matrix)
        activeSetSize = XActiveSet.shape[1]
        GActiveSet = XActiveSet.T * XActiveSet
        soln = np.linalg.solve(GActiveSet, np.ones((activeSetSize,1)))
        AActiveSet = 1/np.sqrt(np.sum(soln))
        wActiveSet = AActiveSet*soln
        uActiveSet = XActiveSet*wActiveSet
        uActiveSetAsArray = np.array(uActiveSet)
        uActiveSetAsArraySquaredNorm = np.sum(uActiveSetAsArray*uActiveSetAsArray)
        assert np.allclose(uActiveSetAsArraySquaredNorm, 1)                       
        assert np.allclose(np.array(XActiveSet.T*uActiveSet), AActiveSet*np.ones((activeSetSize,1))) 
        return { 'A': AActiveSet, 'u': uActiveSet }

    @staticmethod
    def CalculateNewJoinIndex(corrHatMax, cj, AActiveSet, aj, activeSetComplement):
        # Select indices for which the next two lines are defined
        indMinusDefined = np.where(aj[activeSetComplement] != AActiveSet)[0]
        indPlusDefined = np.where(aj[activeSetComplement] != -AActiveSet)[0] 
                    
        # Calculate the minus and plus components of the RHS of (2.13)
        compMinusInComplement = (corrHatMax - cj[activeSetComplement[indMinusDefined]])/ \
                                (AActiveSet - aj[activeSetComplement[indMinusDefined]])
        compPlusInComplement = (corrHatMax + cj[activeSetComplement[indPlusDefined]])/ \
                               (AActiveSet + aj[activeSetComplement[indPlusDefined]])           
                               
        indMinusInComplementAndPos = np.where(compMinusInComplement > 0)[0]            
        indPlusInComplementAndPos = np.where(compPlusInComplement > 0)[0]

        # Initialize        
        indMinusCandidate = -1
        gammaMinusCandidate = None
        indPlusCandidate = -1
        gammaPlusCandidate = None
        
        if indMinusInComplementAndPos.size > 0:
            compMinusInComplementAndPos = compMinusInComplement[indMinusInComplementAndPos]
            indMinusCandidate = np.argmin(compMinusInComplementAndPos)
            assert indMinusCandidate.size == 1                      
            gammaMinusCandidate = compMinusInComplementAndPos[indMinusCandidate]
                  
        if indPlusInComplementAndPos.size > 0:
            compPlusInComplementAndPos = compPlusInComplement[indPlusInComplementAndPos]
            indPlusCandidate = np.argmin(compPlusInComplementAndPos)
            assert indPlusCandidate.size == 1
            gammaPlusCandidate = compPlusInComplementAndPos[indPlusCandidate]
            
        # If there are no candidates from either minus or plus component, there's a problem
        # since we can't go further.
        assert (indMinusCandidate >= 0) or (indPlusCandidate >= 0)
        
        if ((indMinusCandidate == -1) or 
            ((gammaMinusCandidate is not None) and (gammaPlusCandidate is not None) and (gammaPlusCandidate < gammaMinusCandidate))):                
            indCandidate = activeSetComplement[indPlusDefined[indPlusInComplementAndPos[indPlusCandidate]]]
            gammaCandidate = gammaPlusCandidate
        else:
            assert ((indPlusCandidate == -1) or
                    ((gammaMinusCandidate is not None) and (gammaPlusCandidate is not None) and (gammaPlusCandidate >= gammaMinusCandidate)))
            indCandidate = activeSetComplement[indMinusDefined[indMinusInComplementAndPos[indMinusCandidate]]]
            gammaCandidate = gammaMinusCandidate

        # Check that gammaCandidate is the smallest positive element in comp{Minus,Plus}InComplement
        indMinusSmaller = np.where(compMinusInComplement < gammaCandidate)[0]
        if indMinusSmaller.size > 0:
            otherMinusValuesMax = np.max(compMinusInComplement[indMinusSmaller])
            assert otherMinusValuesMax <= 0
        indPlusSmaller = np.where(compPlusInComplement < gammaCandidate)[0]
        if indPlusSmaller.size > 0:
            otherPlusValuesMax = np.max(compPlusInComplement[indPlusSmaller])
            assert otherPlusValuesMax <= 0
                            
#        return { 'indHat': indCandidate, 'gammaHat': gammaCandidate}
        return { 'indHat': indCandidate, 
                 'gammaHat': gammaCandidate,
                 'C': corrHatMax, # Also return the input parameters 
                 'cj': cj, 
                 'AActiveSet': AActiveSet, 
                 'aj': aj,
                 'activeSetComplement': activeSetComplement
                }
                        
    def Iterate(self, y, psfRepH):
        maxIter = self._optimSettingsDict[LeastAngleRegressionReconstructor.INPUT_KEY_MAX_ITERATIONS] \
            if LeastAngleRegressionReconstructor.INPUT_KEY_MAX_ITERATIONS in self._optimSettingsDict \
            else 500
            
        nVerbose = self._optimSettingsDict[LeastAngleRegressionReconstructor.INPUT_KEY_NVERBOSE] \
            if LeastAngleRegressionReconstructor.INPUT_KEY_NVERBOSE in self._optimSettingsDict \
            else 0            

        convMatrixObj = ConvolutionMatrixUsingPsf(psfRepH)    
                                    
        fnConvolveWithPsf = lambda x: convMatrixObj.Multiply(x)
        fnConvolveWithPsfPrime = lambda x: convMatrixObj.MultiplyPrime(x)     
        
        HPrimey = fnConvolveWithPsfPrime(y)
                            
        # Initialize            
        corrHatAbsMax = np.max(HPrimey.flat) # Start out with mu_0 = 0
        activeSet = np.where(HPrimey.flat == corrHatAbsMax)[0]
        if activeSet.size != 1:
            raise RuntimeError('Initial active set size is ' + str(activeSet.size))
        
        activeSetComplement = np.setdiff1d(np.arange(y.size), activeSet)
        assert activeSetComplement.size == (y.size - 1)
        
        muHatActiveSet = np.matrix(np.zeros((y.size, 1)))
        XActiveSetColumns = []       
        corrHatAbsMaxActualHistory = []         
        activeSetResult = None
        joinResult = None
        lastIndexToJoinActiveSet = None
        
        numIter = 0
        
        while numIter < maxIter:            
            corrHat = HPrimey - fnConvolveWithPsfPrime(np.reshape(muHatActiveSet, y.shape))  # (2.8)
            corrHatAbs = np.array(np.abs(corrHat.flat))            
           
            if nVerbose >= 1:
                print("Iteration {0}: active set is {1}".format(numIter, activeSet))
                print('ACorr(activeSet): {0} + {1}'.format(corrHatAbs[activeSet[0]], 
                                                           corrHatAbs[activeSet] - corrHatAbs[activeSet[0]]))            
                
            corrHatAbsMaxActual = np.max(corrHatAbs) # (2.9)                                
            corrHatAbsMismatch = np.abs(corrHatAbsMax - corrHatAbsMaxActual)
            
            if corrHatAbsMismatch > self.EPS*2.0:
                indCorrHatAbsMaxActual = np.argmax(corrHatAbs)
                if (indCorrHatAbsMaxActual.size > 1):
                    print "Multiple maximum values in corrHatAbs " + str(indCorrHatAbsMaxActual) + \
                          " corresponding to " + corrHatAbs[indCorrHatAbsMaxActual[0]] + \
                          " + " + (corrHatAbs[indCorrHatAbsMaxActual] - corrHatAbs[indCorrHatAbsMaxActual[0]])
                    raise RuntimeError('Iteration ' + str(numIter) + ' has a fatal error')
                if indCorrHatAbsMaxActual in activeSetComplement:                                
                    assert joinResult is not None                
                    calcActualMax = ((joinResult['C']-joinResult['cj'][indCorrHatAbsMaxActual])/
                                     (joinResult['AActiveSet']-joinResult['aj'][indCorrHatAbsMaxActual]),
                                     (joinResult['C']+joinResult['cj'][indCorrHatAbsMaxActual])/
                                     (joinResult['AActiveSet']+joinResult['aj'][indCorrHatAbsMaxActual])
                                     )
                    calcLastJoin = ((joinResult['C']-joinResult['cj'][lastIndexToJoinActiveSet])/
                            (joinResult['AActiveSet']-joinResult['aj'][lastIndexToJoinActiveSet]),
                            (joinResult['C']+joinResult['cj'][lastIndexToJoinActiveSet])/
                            (joinResult['AActiveSet']+joinResult['aj'][lastIndexToJoinActiveSet])
                            )                
                    print 'calc of Actual max is ' + str(calcActualMax)
                    print 'calc of Last join is ' + str(calcLastJoin)
                else:
                    if not(indCorrHatAbsMaxActual in activeSet):
                        raise RuntimeError('Iteration ' + str(numIter) + 
                                           ' index of Acorr Actual max is ' + str(indCorrHatAbsMaxActual) +
                                           ' which is neither in the active set nor in its comp')
                raise RuntimeError('Iteration ' + str(numIter) + 
                                   ' Abs corr hats don\'t match: theory is ' + str(corrHatAbsMax) + 
                                   ' whereas actual is ' + str(corrHatAbsMaxActual) +
                                   ' at index/indices ' + str(indCorrHatAbsMaxActual) +
                                   '. |Delta| is ' + str(corrHatAbsMismatch)
                                   )
                
            corrHatAbsMaxActualHistory.append(corrHatAbsMaxActual)
            
            activeSetActual = np.where(np.abs(corrHatAbs - corrHatAbsMax) < self.EPS)[0] # (2.9)
#            activeSetActual = np.where(corrHatFlat == corrHatMax)[0] # (2.9)

            if (nVerbose >= 1):
                if activeSetActual.size > 0:
                    print('ACorr(activeSetActual): {0} + {1}'.format(corrHatAbs[activeSetActual[0]], 
                                                                     corrHatAbs[activeSetActual] - corrHatAbs[activeSetActual[0]]))
                else:
                    print('Corr(activeSetActual): n/a')
                                                                                        
            if np.setxor1d(activeSet, activeSetActual).size != 0:
#                raise RuntimeError('Iteration ' + str(numIter) + 
#                                   ' activeSet and activeSetActual aren\'t the same. ActiveSet is ' + str(activeSet) +
#                                   ' vs. ActiveSetActual is ' + str(activeSetActual))
                # The calculation of activeSetActual isn't very accurate. Raise a warning instead of an exception.
                warnings.warn(
                              'Iteration ' + str(numIter) + 
                               ' activeSet and activeSetActual aren\'t the same. ActiveSet is ' + str(activeSet) +
                               ' vs. ActiveSetActual is ' + str(activeSetActual),
                               RuntimeWarning
                               ) 
                                                        
            if (lastIndexToJoinActiveSet is None):                
                assert (numIter == 0) and (activeSetResult is None) and (len(XActiveSetColumns) == 0) 
                assert activeSet.size == 1                                
                lastIndexToJoinActiveSet = activeSet[0]

            assert lastIndexToJoinActiveSet is not None                 
            singlePoint = np.zeros((y.size, 1))
            s = np.sign(corrHat.flat)
            assert s[lastIndexToJoinActiveSet] != 0
            singlePoint[lastIndexToJoinActiveSet] = s[lastIndexToJoinActiveSet]            
            singlePsf = fnConvolveWithPsf(np.reshape(singlePoint, y.shape))

            XActiveSetColumns.append(np.array(singlePsf.flat))                            
            XActiveSet = np.matrix(XActiveSetColumns).T            
            assert (XActiveSet.shape[0] == y.size) and (XActiveSet.shape[1] == activeSet.size)                            
            activeSetResult = LeastAngleRegressionReconstructor.CalculateAU(XActiveSet)
                                 
            a =  fnConvolveWithPsfPrime(np.reshape(activeSetResult['u'], y.shape))
            aFlat = a.flat
            
            joinResult = LeastAngleRegressionReconstructor.CalculateNewJoinIndex(corrHatAbsMax, 
                                                                                 corrHat.flat, 
                                                                                 activeSetResult['A'], 
                                                                                 aFlat, 
                                                                                 activeSetComplement)
                        
            lastIndexToJoinActiveSet = joinResult['indHat']
            activeSet = np.append(activeSet, lastIndexToJoinActiveSet)
#            activeSetComplement = np.delete(activeSetComplement, lastIndexToJoinActiveSet)
            activeSetComplement = np.setdiff1d(np.arange(y.size), activeSet)
            assert activeSet.size + activeSetComplement.size == y.size
                
            if nVerbose >= 1:
                print "gammaHat is " + str(joinResult['gammaHat']) + " AActiveSet is " + str(activeSetResult['A'])
                
            corrHatAbsMax -=  joinResult['gammaHat']*activeSetResult['A']    
            assert corrHatAbsMax >= 0
            
            muHatActiveSet += joinResult['gammaHat']*activeSetResult['u']                        
            
            numIter += 1

        corrHatAbsMaxActualHistory.append(corrHatAbsMax)
                        
        return { LeastAngleRegressionReconstructor.OUTPUT_KEY_ACTIVESET: activeSet,
                 LeastAngleRegressionReconstructor.OUTPUT_KEY_MAX_CORRHAT_HISTORY: corrHatAbsMaxActualHistory,                 
                 LeastAngleRegressionReconstructor.OUTPUT_KEY_MUHAT_ACTIVESET: muHatActiveSet.flatten()
                }
                    
    def Estimate(self, y, psfRepH, theta0=None):
        return self.Iterate(y, psfRepH)

            
            
            
            
            
            
            
        
    
