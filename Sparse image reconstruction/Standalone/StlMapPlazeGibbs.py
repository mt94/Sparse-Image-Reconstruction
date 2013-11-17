#!/c/Python27/python

import Examples.MapPlazeGibbsSampleReconstructorOnExample as ReconExample
import multiprocessing as mp
import StlConstants

def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])

def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])

if __name__ == "__main__": 
    
    runArgs = [(1000, 300), StlConstants.EXPERIMENT_DESC, StlConstants.IMAGETYPE, StlConstants.IMAGESHAPE, StlConstants.SNRDB, StlConstants.NUM_NONZERO]
    
    fmtString = "Iter. param: ({0},{1}), perf. criteria: {2}/{3}/{4}, timing={5:g}s."

    pool = mp.Pool(processes = StlConstants.NUMPROC)
    resultPool = pool.map(RunReconstructor_12, [runArgs] * StlConstants.NUMTASKS)
    
    with open('result.txt', 'w+') as fh:
        for aResult in resultPool:
            opString = fmtString.format(
                                        runArgs[0][0], runArgs[0][1],
                                        aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                                        aResult['timing_ms'] / 1.0e3                               
                                        )
            fh.write(opString + '\n')
                                                    