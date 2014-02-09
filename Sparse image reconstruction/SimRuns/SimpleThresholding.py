#!/c/Python27/python

import Examples.SimpleThresholdingReconstructorExample as ReconExample
import multiprocessing as mp
import StlConstants

def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])

def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])

if __name__ == '__main__':
    
    """ 
    Run a comparison between standard Landweber iterations and iterations
    with a non-negative thresholding operation.
    """        
    
    runArgsLw = ['landweber', 5e5, StlConstants.EXPERIMENT_DESC, StlConstants.IMAGETYPE, StlConstants.IMAGESHAPE, StlConstants.SNRDB, StlConstants.NUM_NONZERO]
    runArgsLwNneg = ['landweber_nonneg', 5e5, StlConstants.EXPERIMENT_DESC, StlConstants.IMAGETYPE, StlConstants.IMAGESHAPE, StlConstants.SNRDB, StlConstants.NUM_NONZERO]    
            
    pool = mp.Pool(processes = StlConstants.NUMPROC)
    
    resultPool = pool.map(RunReconstructor_12, [runArgsLw, runArgsLwNneg] * StlConstants.NUMTASKS)
    
    fmtString = "{0}: perf. criteria={1}/{2}/{3}, timing={4:g}s. {5}"
    
    with open('result.txt', 'w+') as fh:
        for aResult in resultPool:
            opString = fmtString.format(             
                                        aResult['estimator'],       
                                        aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],                                                                                                                                            
                                        aResult['timing_ms'] / 1.0e3,
                                        aResult['termination_reason']
                                        )
            fh.write(opString + '\n')    
        
