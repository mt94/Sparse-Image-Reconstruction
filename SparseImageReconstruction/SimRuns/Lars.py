#!/c/Python27/python

from ..Examples import LarsReconstructorOnExample as ReconExample
import multiprocessing as mp
from .Constants import (EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO,
                        NUMPROC, NUMTASKS)


def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])

def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])

if __name__ == "__main__":
    RECONSTRUCTOR_DESC = 'lars_lasso'
    MAX_LARS_ITERATIONS = 30
    
    runArgs = [RECONSTRUCTOR_DESC, MAX_LARS_ITERATIONS, EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO]
        
    fmtString = "Best index: {0}/{1}, perf. criteria: {2}/{3}/{4}, timing={5:g}s."
     
    pool = mp.Pool(processes = NUMPROC)
    resultPool = pool.map(RunReconstructor_12, [runArgs] * NUMTASKS)
    
    with open('result.txt', 'w+') as fh:
        for aResult in resultPool:
            opString = fmtString.format(
                                        aResult['ind_best'], MAX_LARS_ITERATIONS,                               
                                        aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                                        aResult['timing_ms'] / 1.0e3                               
                                        )
            fh.write(opString + '\n')