import numpy as np
import warnings

from LarsConstants import LarsConstants
from LarsIterationEvaluator import LarsIterationEvaluator
from Recon.AbstractReconstructor import AbstractReconstructor
from Recon.AbstractIterationsObserver import AbstractIterationsObserver
from Systems.AbstractConvolutionMatrix import AbstractConvolutionMatrix
#from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

class LarsReconstructor(AbstractReconstructor):
    
    def __init__(self, optimSettingsDict=None):   
        super(LarsReconstructor, self).__init__()                        
        self.EPS = optimSettingsDict.get(LarsConstants.INPUT_KEY_EPS, 1.0e-7)                    
        self._optimSettingsDict = optimSettingsDict            

    """ Calculate AActiveSet and uActiveSet using XActiveSet via (2.5) and (2.6). Assume that 
        XActiveSet is a matrix. uActiveSet is the equiangular vector. """
    @staticmethod    
    def _CalculateActiveSetVariables(XActiveSet):
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
        return { 'A': AActiveSet, 'u': uActiveSet, 'w': wActiveSet }

    @staticmethod
    def _CalculateNewJoinIndex(corrHatAbsMax, cj, AActiveSet, aj, activeSetComplement, EPS):
        # Select indices for which the next two lines are defined
        indMinusDefined = np.where(aj[activeSetComplement] != AActiveSet)[0]
        indPlusDefined = np.where(aj[activeSetComplement] != -AActiveSet)[0] 
                    
        # Calculate the minus and plus components of the RHS of (2.13)
        compMinusInComplement = (corrHatAbsMax - cj[activeSetComplement[indMinusDefined]])/ \
                                (AActiveSet - aj[activeSetComplement[indMinusDefined]])
        compPlusInComplement = (corrHatAbsMax + cj[activeSetComplement[indPlusDefined]])/ \
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

        calcPair = ((corrHatAbsMax - cj[indCandidate])/(AActiveSet - aj[indCandidate]),
                    (corrHatAbsMax + cj[indCandidate])/(AActiveSet + aj[indCandidate]))
        assert np.min(calcPair) == gammaCandidate
                    
        # The problem now is that, since floating point accuracy has a limit, have to be 
        # careful of other values are too close to our supposed minimum.                
        indCandidateOther = np.array([]);
        gammaCandidateOtherDelta = np.array([]);
        
        indMinusClose = np.where(np.abs(compMinusInComplement - gammaCandidate) < EPS*2)[0]
        if indMinusClose.size > 0:
            # Recall that compMinusInComplement <- compMinus[activeSetComplement[indMinusDefined]]
            indTmp = activeSetComplement[indMinusDefined[indMinusClose]]
            bPresent = np.in1d(np.array(indCandidate), indTmp)
            if bPresent[0]:
                indTmp = np.setxor1d(indTmp, np.array(indCandidate))
            if indTmp.size > 0:                        
                indCandidateOther = np.append(indCandidateOther, indTmp)
                gammaCandidateOtherDelta = np.append(gammaCandidateOtherDelta, 
                                                     (corrHatAbsMax - cj[indTmp])/(AActiveSet - aj[indTmp]) - gammaCandidate
                                                     )            
            
        indPlusClose = np.where(np.abs(compPlusInComplement - gammaCandidate) < EPS*2)[0]
        if indPlusClose.size > 0:            
            # Recall that compPlusInComplement <- compPlus[activeSetComplement[indPlusDefined]]
            indTmp = activeSetComplement[indPlusDefined[indPlusClose]]
            bPresent = np.in1d(np.array(indCandidate), indTmp)
            if bPresent[0]:
                indTmp = np.setxor1d(indTmp, np.array(indCandidate))            
            if indTmp.size > 0:
                indCandidateOther = np.append(indCandidateOther, indTmp)
                gammaCandidateOtherDelta = np.append(gammaCandidateOtherDelta,
                                                     (corrHatAbsMax + cj[indTmp])/(AActiveSet + aj[indTmp]) - gammaCandidate
                                                     )
                                                
        return { 'indHat': indCandidate, 
                 'gammaHat': gammaCandidate,
                 'indHatOther': indCandidateOther,
                 'gammaHatOtherDelta': gammaCandidateOtherDelta,
                 'C': corrHatAbsMax, # Also return the input parameters 
                 'cj': cj, 
                 'AActiveSet': AActiveSet, 
                 'aj': aj,
                 'activeSetComplement': activeSetComplement
                }

    """ To be called when _CalculateNewJoinIndex returns several {ind,gamma}Candidate values """
    @staticmethod
    def _SelectNewJoinIndex(joinResultDict, activeSetResultDict, activeSet, corrHatAbsMax, muHatActiveSet, fnComputeCorrHat, EPS):
        assert ('indHat' in joinResultDict) and ('indHatOther' in joinResultDict) and (joinResultDict['indHatOther'].size > 0)        
                
        indHatList = np.append(joinResultDict['indHatOther'], joinResultDict['indHat'])        
        indHatIntList = []
                
        for indHat in indHatList:
            indHatInt = np.int(indHat)
            indHatIntList.append(indHatInt)
                                    
            calcPair = ((corrHatAbsMax - joinResultDict['cj'][indHatInt])/(joinResultDict['AActiveSet'] - joinResultDict['aj'][indHatInt]),
                        (corrHatAbsMax + joinResultDict['cj'][indHatInt])/(joinResultDict['AActiveSet'] + joinResultDict['aj'][indHatInt]))
            gammaHat = np.min(calcPair)
            if gammaHat <= 0:
                gammaHat = np.max(calcPair)
                assert gammaHat > 0
                
            # Update variables                
            corrHatAbsMaxNext = corrHatAbsMax - gammaHat*activeSetResultDict['A']            
            muHatActiveSetNext = muHatActiveSet + gammaHat*activeSetResultDict['u']
        
            # Compute next set of correlations
            corrHatNext = fnComputeCorrHat(muHatActiveSetNext)
            corrHatNextAbsMismatch = np.abs(corrHatAbsMaxNext - np.max(np.abs(corrHatNext.flat)))
            
            outputDict = { 
                          'indHat': str(indHatInt), 'calcPair': str(calcPair), 'corrHatNextAbsMismatch': str(corrHatNextAbsMismatch),
                          'cj': str(joinResultDict['cj'][indHatInt]), 'aj': str(joinResultDict['aj'][indHatInt])
                         }
            debugMsg = ' '.join([key + '=' + outputDict[key] for key in outputDict])
            
#        return acceptableJoinIndex
#        raise RuntimeError('Abort')
#        return joinResultDict['indHat']
        return np.array(indHatIntList), debugMsg

    def _GetVariablesForIteration(self, y, fnConvolveWithPsfPrime):
        maxIter = self._optimSettingsDict.get(LarsConstants.INPUT_KEY_MAX_ITERATIONS, 500)
        nVerbose = self._optimSettingsDict.get(LarsConstants.INPUT_KEY_NVERBOSE, 0)
        bEnforceOneatatimeJoin = self._optimSettingsDict.get(LarsConstants.INPUT_KEY_ENFORCE_ONEATATIME_JOIN, False)

        HPrimey = fnConvolveWithPsfPrime(y)        
        fnComputeCorrHat = lambda x: HPrimey - fnConvolveWithPsfPrime(np.reshape(x, y.shape)) # 2.8

        # Initialize            
        corrHatAbsMax = np.max(HPrimey.flat) # Start out with mu_0 = 0
        activeSet = np.where(HPrimey.flat == corrHatAbsMax)[0]
        if activeSet.size != 1:
            raise RuntimeError('Initial active set size is ' + str(activeSet.size))
        
        activeSetComplement = np.setdiff1d(np.arange(y.size), activeSet)
        assert activeSetComplement.size == (y.size - 1)
                
        iterObserver = self._optimSettingsDict.get(LarsConstants.INPUT_KEY_ITERATIONS_OBSERVER)            
        if iterObserver is not None:
            assert isinstance(iterObserver, AbstractIterationsObserver)
            
        return maxIter, nVerbose, bEnforceOneatatimeJoin, fnComputeCorrHat, corrHatAbsMax, activeSet, activeSetComplement, iterObserver
                                            
    def Iterate(self, y, fnConvolveWithPsf, fnConvolveWithPsfPrime):
                                            
        maxIter, nVerbose, bEnforceOneatatimeJoin, fnComputeCorrHat, corrHatAbsMax, activeSet, activeSetComplement, iterObserver = \
            self._GetVariablesForIteration(y, fnConvolveWithPsfPrime)
                                                                                
        muHatActiveSet = np.matrix(np.zeros((y.size, 1)))
        betaHatActiveSet = np.matrix(np.zeros((y.size, 1)))
        
        XActiveSetColumns = []       
        corrHatAbsMaxActualHistory = []                
        activeSetResult = None
        joinResult = None
        
        assert activeSet.size == 1                                
        lastIndexToJoinActiveSet = np.array([activeSet[0]])        
        
        numIter = 0
        
        while numIter < maxIter:            
#            corrHat = HPrimey - fnConvolveWithPsfPrime(np.reshape(muHatActiveSet, y.shape))  # (2.8)
            corrHat = fnComputeCorrHat(muHatActiveSet)
            corrHatAbs = np.array(np.abs(corrHat.flat))
            msgBuffer = []            
           
            if nVerbose >= 1:
                msgBuffer.append("=== Iteration {0}: active set is {1} ===".format(numIter, activeSet))
                msgBuffer.append('ACorr(activeSet): {0} + {1}'.format(corrHatAbs[activeSet[0]], 
                                                                      corrHatAbs[activeSet] - corrHatAbs[activeSet[0]]))            
                
            corrHatAbsMaxActual = np.max(corrHatAbs) # (2.9)                                
            corrHatAbsMismatch = np.abs(corrHatAbsMax - corrHatAbsMaxActual)
                        
            if corrHatAbsMismatch > self.EPS*2:
                indCorrHatAbsMaxActual = np.argmax(corrHatAbs)
                raise RuntimeError('Iteration {0}: Abs corr hats don\'t match: theory is {1} whereas actual is {2} @ index/indices {3}, |delta| is {4}'.format(numIter,
                                                                                                                                                               corrHatAbsMax,
                                                                                                                                                               corrHatAbsMaxActual,
                                                                                                                                                               indCorrHatAbsMaxActual,
                                                                                                                                                               corrHatAbsMismatch)
                                   )
                
            corrHatAbsMaxActualHistory.append(corrHatAbsMaxActual)
            
            activeSetActual = np.where(np.abs(corrHatAbs - corrHatAbsMax) < self.EPS)[0] # (2.9)

            if (nVerbose >= 1):
                if activeSetActual.size > 0:
                    msgBuffer.append('ACorr(activeSetActual): {0} + {1}'.format(corrHatAbs[activeSetActual[0]], 
                                                                                corrHatAbs[activeSetActual] - corrHatAbs[activeSetActual[0]]))
                else:
                    msgBuffer.append('Corr(activeSetActual): n/a')
                                                                                        
            if np.setxor1d(activeSet, activeSetActual).size != 0:
                # Numerical inaccuracy might result in a difference
                warnings.warn("Iteration {0}: activeSet and activeSetActual aren't the same.\n".format(numIter) +
                              "ActiveSet is {0} vs. ActiveSetActual is {1}".format(activeSet, activeSetActual),
                              RuntimeWarning
                              ) 
                                                                                                                
            corrHatSign = np.sign(corrHat.flat)                                    
            
            assert lastIndexToJoinActiveSet is not None
            
            for jHat in lastIndexToJoinActiveSet:                            
                singlePoint = np.zeros(corrHatSign.shape)
                assert corrHatSign[jHat] != 0                        
                singlePoint[jHat] = corrHatSign[jHat]            
                singlePsf = fnConvolveWithPsf(np.reshape(singlePoint, y.shape))
                XActiveSetColumns.append(np.array(singlePsf.flat))             
                           
            XActiveSet = np.matrix(XActiveSetColumns).T
             
            if (XActiveSet.shape[0] != y.size) or (XActiveSet.shape[1] != activeSet.size):
                raise RuntimeError('Expect XActiveSet.shape={0}x{1} to equal {2}x{3}'.format(XActiveSet.shape[0],
                                                                                             XActiveSet.shape[1],
                                                                                             y.size,
                                                                                             activeSet.size)
                                   )                    
                                           
            activeSetResult = LarsReconstructor._CalculateActiveSetVariables(XActiveSet)
                                 
            a =  fnConvolveWithPsfPrime(np.reshape(activeSetResult['u'], y.shape))
            aFlat = a.flat
            
            joinResult = LarsReconstructor._CalculateNewJoinIndex(corrHatAbsMax, 
                                                                  corrHat.flat, 
                                                                  activeSetResult['A'], 
                                                                  aFlat, 
                                                                  activeSetComplement,
                                                                  self.EPS,
                                                                  )
            
            if joinResult['gammaHatOtherDelta'].size > 0:
                msgOutput = 'gammaHat is {0} while the delta is {1}\n'.format(joinResult['gammaHat'], joinResult['gammaHatOtherDelta']) + \
                            'indHat is {0} while the other possibilities are {1}'.format(joinResult['indHat'], joinResult['indHatOther'])
                if bEnforceOneatatimeJoin:
                    raise RuntimeError(msgOutput)
                else:
                    warnings.warn(msgOutput, RuntimeWarning)
                lastIndexToJoinActiveSet, debugMsg = LarsReconstructor._SelectNewJoinIndex(
                                                                                           joinResult, 
                                                                                           activeSetResult, 
                                                                                           activeSet, 
                                                                                           corrHatAbsMax, 
                                                                                           muHatActiveSet, 
                                                                                           fnComputeCorrHat, 
                                                                                           self.EPS
                                                                                           )  
                if nVerbose >= 1:
                    msgBuffer.append(debugMsg)
                for jHat in lastIndexToJoinActiveSet:
                    activeSet = np.append(activeSet, np.int(jHat))
            else:
                lastIndexToJoinActiveSet = np.array([np.int(joinResult['indHat'])])
                activeSet = np.append(activeSet, np.int(joinResult['indHat']))
               
            activeSetComplement = np.setdiff1d(np.arange(y.size), activeSet)
            assert activeSet.size + activeSetComplement.size == y.size
                                
            corrHatAbsMax -=  joinResult['gammaHat']*activeSetResult['A']    
            assert corrHatAbsMax >= 0

            if nVerbose >= 1:
                msgBuffer.append("gammaHat is {0}, AActiveSet is {1}, predicted corrHatAbsMax is is {2}".format(joinResult['gammaHat'],
                                                                                                                activeSetResult['A'],
                                                                                                                corrHatAbsMax 
                                                                                                                )
                                 )
                
            muHatActiveSet += joinResult['gammaHat']*activeSetResult['u']
                                                        
            dHat = np.zeros(betaHatActiveSet.shape)
            dHat[activeSet] = np.transpose(corrHatSign[activeSet]*np.transpose(activeSetResult['w']))
            betaHatActiveSet += joinResult['gammaHat']*dHat
                    
            if iterObserver is not None:
                iterObserver.UpdateState({
                                          LarsIterationEvaluator.STATE_KEY_THETA: betaHatActiveSet,
                                          LarsIterationEvaluator.STATE_KEY_FIT_ERROR: y - fnConvolveWithPsf(np.reshape(betaHatActiveSet, y.shape)),
                                          LarsIterationEvaluator.OUTPUT_METRIC_CORRHATABS_MAX: corrHatAbsMax
                                          })  
                                    
            numIter += 1
            
            if len(msgBuffer) > 0:
                print msgBuffer[0]
                for msgLine in msgBuffer[1:]:
                    print "   " + msgLine

        corrHatAbsMaxActualHistory.append(corrHatAbsMax)
                        
        return { LarsConstants.OUTPUT_KEY_ACTIVESET: activeSet,
                 LarsConstants.OUTPUT_KEY_MAX_CORRHAT_HISTORY: corrHatAbsMaxActualHistory,                 
                 LarsConstants.OUTPUT_KEY_MUHAT_ACTIVESET: muHatActiveSet.flatten()
                }
                    
    def Estimate(self, y, convMatrixObj, theta0=None):
        assert isinstance(convMatrixObj, AbstractConvolutionMatrix)      
        return self.Iterate(y,                             
                            lambda x: convMatrixObj.Multiply(x),
                            lambda x: convMatrixObj.MultiplyPrime(x)
                            )

        
    
