#!/c/Python27/python

import Examples.LarsReconstructorOnExample as ReconExample
import multiprocessing as mp

def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])

def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])

if __name__ == "__main__":
    RECONSTRUCTOR_DESC = 'lars_lasso'
    MAX_LARS_ITERATIONS = 30
    EXPERIMENT_DESC = 'mrfm2d'
    IMAGETYPE = 'random_discrete'
    IMAGESHAPE = (32, 32); #(32, 32, 14)
    SNRDB = 2
    NUM_NONZERO = 16
    
    runArgs = [RECONSTRUCTOR_DESC, MAX_LARS_ITERATIONS, EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO]
    
    NUMPROC = 4
    NUMTASKS = 30
    
    fmtString = "Best index: {0}/{1}, perf. criteria: {2}/{3}/{4}, timing={5:g}s."
     
    pool = mp.Pool(processes=NUMPROC)
    resultPool = pool.map(RunReconstructor_12, [runArgs] * NUMTASKS)
    
    for aResult in resultPool:
        print(fmtString.format(
                               aResult['ind_best'], MAX_LARS_ITERATIONS,                               
                               aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                               aResult['timing_ms'] / 1.0e3                               
                               ))