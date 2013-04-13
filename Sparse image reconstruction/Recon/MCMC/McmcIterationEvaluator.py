import numpy as np
import pylab as plt

from Recon.AbstractIterationsObserver import AbstractIterationsObserver
from Systems.NumericalHelper import NumericalHelper

class McmcIterationEvaluator(AbstractIterationsObserver):
    """
    Observer for MCMC methods
    """
    
    STATE_KEY_COUNT_ITER = 'count_iter'
    STATE_KEY_X_ITER = 'x_iter'
    STATE_KEY_W_ITER = 'w_iter'
    STATE_KEY_A_ITER = 'a_iter'
    STATE_KEY_NOISEVAR_ITER = 'noisevar_iter'
        
    def __init__(self, Eps, xShape, xTrue=None, xFigureNum=None, countIterationDisplaySet=None):
        super(McmcIterationEvaluator, self).__init__()
        
        self.bVerbose = True
        self.Eps = Eps
        self.xShape = xShape
        self.xTrue = xTrue        
                    
        if (xFigureNum is not None) and (xFigureNum > 0):
            self.xFigureNum = xFigureNum
        else:
            self.xFigureNum = -1
            
        self.countIteration = 0
        
        if countIterationDisplaySet is not None:
            self.countIterationDisplaySet = countIterationDisplaySet
        else:
            # Default value
            self.countIterationDisplaySet = np.append(1, np.arange(100, 4000, 100))
                
        self.xHistory = [];
        self.wHistory = [];
        self.aHistory = [];
        self.noiseVarHistory = [];       
    
    """ Implementation of abstract methods """
    
    @property
    def TerminateIterations(self):
        return False # Never terminate
    
    def UpdateEstimates(self, thetaNp1, thetaN, fitErrorN):
        raise NotImplementedError('Method unimplemented')
    
    def UpdateState(self, stateDict):
        if McmcIterationEvaluator.STATE_KEY_COUNT_ITER in stateDict:
            self.countIteration = stateDict[McmcIterationEvaluator.STATE_KEY_COUNT_ITER]
        else:
            self.countIteration += 1    
    
        keysForMsg = (McmcIterationEvaluator.STATE_KEY_W_ITER,
                      McmcIterationEvaluator.STATE_KEY_A_ITER,
                      McmcIterationEvaluator.STATE_KEY_NOISEVAR_ITER)
        
        if self.bVerbose and all(k in stateDict for k in keysForMsg):        
            print("=> Iter {0}: hyper samp.: w={1}, a={2}; var samp.: {3}".format(self.countIteration,
                                                                                  stateDict[McmcIterationEvaluator.STATE_KEY_W_ITER],
                                                                                  stateDict[McmcIterationEvaluator.STATE_KEY_A_ITER],
                                                                                  stateDict[McmcIterationEvaluator.STATE_KEY_NOISEVAR_ITER]))
            
        if (McmcIterationEvaluator.STATE_KEY_X_ITER in stateDict) and isinstance(stateDict[McmcIterationEvaluator.STATE_KEY_X_ITER], np.ndarray):
            xIter = stateDict[McmcIterationEvaluator.STATE_KEY_X_ITER]
            if self.xTrue is None:
                raise ValueError("xTrue member isn't initialized")
            xErr = self.xTrue - xIter[:,0]
            n0Next, n1Next = NumericalHelper.CalculateNumZerosNonzeros(xIter, self.Eps)
            if self.bVerbose:
                print("   Sampled x: n0={0}, |x|_0={1}, |x|_1={2}, |x-xTrue|_1={3}".format(n0Next, 
                                                                                           n1Next,                                                                                           
                                                                                           np.sum(np.abs(xIter)),
                                                                                           np.sum(np.abs(xErr))
                                                                                           )) 
                           
            if self.bVerbose \
                and (self.xShape is not None) \
                and (len(self.xShape) == 2) \
                and (self.xFigureNum > 0) \
                and np.in1d(self.countIteration, self.countIterationDisplaySet)[0] \
                :
                # Display the truth
                plt.figure(1)
                plt.imshow(np.reshape(self.xTrue, self.xShape), interpolation='none')
                plt.colorbar()
                plt.title('Actual theta')
                # Display the sample
                plt.figure(self.xFigureNum)
                plt.imshow(np.reshape(xIter, self.xShape), interpolation='none')
                plt.title("Iter {0}".format(self.countIteration))                    
                plt.show()                  
                        
                
