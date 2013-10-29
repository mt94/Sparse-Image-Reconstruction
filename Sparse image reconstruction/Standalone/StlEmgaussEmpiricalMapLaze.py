#!/c/Python27/python

import Examples.EmgaussEmpiricalMapLazeReconstructorOnExample as ReconExample
import math
import multiprocessing as mp

def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])

def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])

if __name__ == "__main__":
    EXPERIMENT_DESC = 'mrfm2d'
    IMAGETYPE = 'random_discrete'
    IMAGESHAPE = (32, 32);  # (32, 32, 14) 
    GSUP = 1/math.sqrt(2)
    SNRDB = 20
    NUM_NONZERO = 16
    
    # For MAP1
    runArgsMap1 = [EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO]        
    # For MAP2
    runArgsMap2 = [EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO, GSUP]
        
    NUMPROC = 4
    NUMTASKS = 30
        
    fmtString = "{0}: est. hyper.={1}, perf. criteria={2}/{3}/{4}, timing={5:g}s. {6}"
    
    pool = mp.Pool(processes=NUMPROC)
    resultPool = pool.map(RunReconstructor_12, [runArgsMap1, runArgsMap2] * NUMTASKS)
    
    with open('result.txt', 'w+') as fh:
        for aResult in resultPool:
            opString = fmtString.format(
                                        aResult['reconstructor_desc'],
                                        aResult['hyperparameter'],
                                        aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                                        aResult['timing_ms'] / 1.0e3,
                                        aResult['termination_reason']
                                        )
            fh.write(opString + '\n') 