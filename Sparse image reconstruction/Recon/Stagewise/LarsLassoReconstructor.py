import numpy as np
import warnings

from LarsConstants import LarsConstants
from LarsIterationEvaluator import LarsIterationEvaluator
from LarsReconstructor import LarsReconstructor

class LarsLassoReconstructor(LarsReconstructor):
        
    def __init__(self, optimSettingsDict=None):
        super(LarsLassoReconstructor, self).__init__(optimSettingsDict)
        self._signViolationNumIter = 0

    @staticmethod    
    def _CalculateSignViolationIndex(activeSet, betaHatActiveSet, wActiveSet, corrHatSign):
        dHat = np.zeros(betaHatActiveSet.shape)
        dHat[activeSet] = np.transpose(corrHatSign[activeSet]*np.transpose(wActiveSet))
        gammaValue = -np.array(betaHatActiveSet)/dHat
        indStrictlyPositive = np.where(gammaValue > 0)[0]
        if indStrictlyPositive.size > 0:
            indArgMin = np.argmin(gammaValue[indStrictlyPositive])
            jTilde = indStrictlyPositive[indArgMin]
            gammaTilde = gammaValue[jTilde]
            assert gammaTilde > 0
        else:
            jTilde = None
            gammaTilde = np.Inf
        return { 'indViolation': jTilde,
                 'gammaViolation': gammaTilde,
                 'd': dHat
                }                                     
                
    def Iterate(self, y, fnConvolveWithPsf, fnConvolveWithPsfPrime):
        """
        Override the Iterate method of the base class.
        """
        maxIter, nVerbose, bEnforceOneatatimeJoin, fnComputeCorrHat, corrHatAbsMax, activeSet, activeSetComplement, iterObserver = \
            self._GetVariablesForIteration(y, fnConvolveWithPsfPrime)
        
        # Only support one join at a time for the LARS-LASSO algorithm
        assert bEnforceOneatatimeJoin is True
                                
        muHatActiveSet = np.zeros((y.size, 1))
        betaHatActiveSet = np.zeros((y.size, 1))
        
        XActiveSetColumns = []       
        corrHatAbsMaxActualHistory = []                 
        activeSetResult = None
        joinResult = None
        violationResult = None
        
        assert activeSet.size == 1                                
        lastIndexToJoinActiveSet = np.array([activeSet[0]])   
        lastIndexToLeaveActiveSet = None                
        activeSetIndexToDelete = None
        
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
                        
            if corrHatAbsMismatch > 1.0e-5: # self.EPS*10 is too strict???
                indCorrHatAbsMaxActual = np.argmax(corrHatAbs)
                fmtString = "Iteration {0}: Abs corr hats don't match: theory is {1} whereas actual is {2} @ index/indices {3}, |delta| is {4}"
#                raise RuntimeError(fmtString.format(numIter, corrHatAbsMax, corrHatAbsMaxActual, indCorrHatAbsMaxActual, corrHatAbsMismatch))
                warnings.warn(
                              fmtString.format(numIter, corrHatAbsMax, corrHatAbsMaxActual, indCorrHatAbsMaxActual, corrHatAbsMismatch),
                              RuntimeWarning
                              )
                
            corrHatAbsMaxActualHistory.append(corrHatAbsMaxActual)
            
            activeSetActual = np.where(np.abs(corrHatAbs - corrHatAbsMax) < self.EPS)[0] # (2.9)
#            activeSetActual = np.where(corrHatFlat == corrHatMax)[0] # (2.9)

            if (nVerbose >= 1):
                if activeSetActual.size > 0:
                    msgBuffer.append('ACorr(activeSetActual): {0} + {1}'.format(corrHatAbs[activeSetActual[0]], 
                                                                                corrHatAbs[activeSetActual] - corrHatAbs[activeSetActual[0]]))
                else:
                    msgBuffer.append('Corr(activeSetActual): n/a')
                                                                                        
            if np.setxor1d(activeSet, activeSetActual).size != 0:
                # Numerical inaccuracy might result in a difference                
                warnings.warn(
                              "Iteration {0}: activeSet and activeSetActual aren't the same.\n".format(numIter) +
                              "ActiveSet is {0} vs. ActiveSetActual is {1}".format(activeSet, activeSetActual),
                              RuntimeWarning
                              ) 
                                                        
                                                
            corrHatSign = np.sign(corrHat.flat)            
            
            if lastIndexToJoinActiveSet is not None:            
                assert lastIndexToJoinActiveSet.size == 1 # Consequence of insisting that bEnforceOneatatimeJoin be True                
                jHat = lastIndexToJoinActiveSet[0]                            
                assert corrHatSign[jHat] != 0
                singlePoint = np.zeros(corrHatSign.shape)                        
                singlePoint[jHat] = corrHatSign[jHat]            
                singlePsf = fnConvolveWithPsf(np.reshape(singlePoint, y.shape))
                XActiveSetColumns.append(np.array(singlePsf.flat))        
            elif lastIndexToLeaveActiveSet is not None:
                assert lastIndexToLeaveActiveSet.size == 1                
                assert activeSetIndexToDelete.size == 1
                del XActiveSetColumns[activeSetIndexToDelete[0]]                
            else:
                raise RuntimeError('Iteration {0}: no change to the active set scheduled'.format(numIter))
                           
            XActiveSet = np.matrix(XActiveSetColumns).T
             
            if (XActiveSet.shape[0] != y.size) or (XActiveSet.shape[1] != activeSet.size):
                raise RuntimeError('Expect XActiveSet.shape={0}x{1} to equal {2}x{3}'.format(XActiveSet.shape[0],
                                                                                             XActiveSet.shape[1],
                                                                                             y.size,
                                                                                             activeSet.size)
                                   )                    
                                           
            activeSetResult = LarsReconstructor._CalculateActiveSetVariables(XActiveSet)
                                 
            a =  fnConvolveWithPsfPrime(
                                        np.reshape(
                                                   np.array(activeSetResult['u']), 
                                                   y.shape
                                                   )
                                        )
            aFlat = a.flat
            
            joinResult = LarsReconstructor._CalculateNewJoinIndex(corrHatAbsMax, 
                                                                  corrHat.flat, 
                                                                  activeSetResult['A'], 
                                                                  aFlat, 
                                                                  activeSetComplement,
                                                                  self.EPS,
                                                                  )
            
            violationResult = LarsLassoReconstructor._CalculateSignViolationIndex(activeSet, betaHatActiveSet, activeSetResult['w'], corrHatSign)
            
            if violationResult['gammaViolation'] >= joinResult['gammaHat']:
                # No sign violation, so do a normal LARS iteration
                if joinResult['gammaHatOtherDelta'].size > 0:
                    msgOutput = 'gammaHat is {0} while the delta is {1}\n'.format(joinResult['gammaHat'], joinResult['gammaHatOtherDelta']) + \
                                'indHat is {0} while the other possibilities are {1}'.format(joinResult['indHat'], joinResult['indHatOther'])
                    raise RuntimeError(msgOutput)
                else:
                    lastIndexToJoinActiveSet = np.array([np.int(joinResult['indHat'])])
                    lastIndexToLeaveActiveSet = None
                    activeSet = np.append(activeSet, np.int(joinResult['indHat']))
                    gammaHatCurrentIter = joinResult['gammaHat']
            else:
                # Sign violation
                self._signViolationNumIter += 1
                if nVerbose >= 1:
                    msgBuffer.append('Sign violation: gammaHat is {0} vs. gammaTilde is {1}'.format(joinResult['gammaHat'], violationResult['gammaViolation']))
                if violationResult['indViolation'].size > 1:
                    raise RuntimeError('Lasso sign violation occurs at multiple indices in the active set')
                else:
                    lastIndexToJoinActiveSet = None
                    lastIndexToLeaveActiveSet = violationResult['indViolation']
                    activeSetIndexToDelete = np.where(activeSet == lastIndexToLeaveActiveSet)[0]
                    assert activeSetIndexToDelete.size == 1
                    activeSet = np.delete(activeSet, activeSetIndexToDelete[0])                    
                    gammaHatCurrentIter = violationResult['gammaViolation'][0]
               
            activeSetComplement = np.setdiff1d(np.arange(y.size), activeSet)
            assert activeSet.size + activeSetComplement.size == y.size
                                
            corrHatAbsMax -=  gammaHatCurrentIter*activeSetResult['A']    
            assert corrHatAbsMax >= 0

            if nVerbose >= 1:
                msgBuffer.append("gammaHat is {0}, AActiveSet is {1}, predicted corrHatAbsMax is is {2}".format(gammaHatCurrentIter,
                                                                                                                activeSetResult['A'],
                                                                                                                corrHatAbsMax 
                                                                                                                )
                                 )
                
            muHatActiveSet += gammaHatCurrentIter*activeSetResult['u']
            betaHatActiveSet += gammaHatCurrentIter*violationResult['d']  
            
            if iterObserver is not None:                                
                iterObserver.UpdateState({
                                          LarsIterationEvaluator.STATE_KEY_THETA: betaHatActiveSet,
                                          LarsIterationEvaluator.STATE_KEY_FIT_ERROR: y - fnConvolveWithPsf(np.reshape(betaHatActiveSet, y.shape)),
                                          LarsIterationEvaluator.OUTPUT_METRIC_CORRHATABS_MAX: corrHatAbsMax,
                                          LarsIterationEvaluator.FUNC_KEY_MU_FROM_THETA: lambda aTheta: fnConvolveWithPsf(np.reshape(aTheta, y.shape))
                                          },
                                         bPlotThetaErr=False
                                         )              
            numIter += 1
            
            if len(msgBuffer) > 0:
                print msgBuffer[0]
                for msgLine in msgBuffer[1:]:
                    print "   " + msgLine

        corrHatAbsMaxActualHistory.append(corrHatAbsMax)
                        
        return { LarsConstants.OUTPUT_KEY_ACTIVESET: activeSet,
                 LarsConstants.OUTPUT_KEY_MAX_CORRHAT_HISTORY: corrHatAbsMaxActualHistory,                 
                 LarsConstants.OUTPUT_KEY_MUHAT_ACTIVESET: muHatActiveSet.flatten(),
                 LarsConstants.OUTPUT_KEY_BETAHAT_ACTIVESET: betaHatActiveSet.flatten(),
                 LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER: self._signViolationNumIter
                }
        
