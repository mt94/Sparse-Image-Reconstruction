#!/c/Python27/python

import gc
import math
import multiprocessing as mp
from time import gmtime, strftime

from ..Examples import MapPlazeGibbsSampleReconstructorOnExample as ReconExample
from .Constants import (
    EXPERIMENT_DESC,
    IMAGETYPE,
    IMAGESHAPE,
    SNRDB,
    NUM_NONZERO,
    NUMTASKS,
)


def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])


def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])


def CreateIterationsVec(numTasks, maxSimultaneousProcesses):
    numIterations = int(math.floor(float(numTasks) / maxSimultaneousProcesses))
    if numIterations * maxSimultaneousProcesses != numTasks:
        numIterations += 1
    n = int(math.floor(float(numTasks) / numIterations))
    processesPerIter = [n for i in range(numIterations)]
    numTasksLeft = numTasks - n * numIterations
    for i in range(numTasksLeft):
        processesPerIter[i] += 1
    return processesPerIter


if __name__ == "__main__":
    fmtString = "Iter. param: ({0},{1}), perf. criteria: {2}/{3}/{4}, timing={5:g}s."
    runArgs = [(1000, 300), EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO]

    with open("result.txt", "w+") as fh:
        fh.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")

    NUMPROC = 2  # Ignore Constants.NUMPROC
    processesPerIter = CreateIterationsVec(NUMTASKS, NUMPROC)

    for cntIter in range(len(processesPerIter)):
        # Run exactly processesPerIter[cntIter] processes, which should be strictly less than NUMPROC
        assert processesPerIter[cntIter] <= NUMPROC, "Exceeding NUMPROC processes"
        pool = mp.Pool(processes=processesPerIter[cntIter], maxtasksperchild=1)
        resultPool = pool.map(
            ReconExample.RunReconstructor, [runArgs] * processesPerIter[cntIter]
        )

        # Append the results to the output file
        with open("result.txt", "a+") as fh:
            for aResult in resultPool:
                opString = fmtString.format(
                    runArgs[0][0],
                    runArgs[0][1],
                    aResult["normalized_l2_error_norm"],
                    aResult["normalized_detection_error"],
                    aResult["normalized_l0_norm"],
                    aResult["timing_ms"] / 1.0e3,
                )
                fh.write(opString + "\n")

        # Clean up
        print("Cleaning up pool...")
        pool.close()
        pool.join()

        del resultPool
        del pool

        # Ask the GC to collect
        print("Calling the gc...")
        gc.collect()
