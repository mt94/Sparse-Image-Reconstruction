#!/c/Python27/python

import multiprocessing as mp
from time import gmtime, strftime

import Examples.MapPlazeGibbsSampleReconstructorOnExample as ReconExample
import Constants

def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])

def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])

if __name__ == "__main__": 
    
    runArgs = [(1000, 300), Constants.EXPERIMENT_DESC, Constants.IMAGETYPE, Constants.IMAGESHAPE, Constants.SNRDB, Constants.NUM_NONZERO]
        
    with open('result.txt', 'w+') as fh:
        fh.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        
    pool = mp.Pool(processes = Constants.NUMPROC)
    resultPool = pool.map(RunReconstructor_pm1, [runArgs] * Constants.NUMTASKS)
    
    fmtString = "Iter. param: ({0},{1}), perf. criteria: {2}/{3}/{4}, timing={5:g}s."
        
    with open('result.txt', 'a+') as fh:
        for aResult in resultPool:
            opString = fmtString.format(
                                        runArgs[0][0], runArgs[0][1],
                                        aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                                        aResult['timing_ms'] / 1.0e3                               
                                        )
            fh.write(opString + '\n')
            