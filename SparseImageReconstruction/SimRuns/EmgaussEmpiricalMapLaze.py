#!/c/Python27/python

from ..Examples import EmgaussEmpiricalMapLazeReconstructorOnExample as ReconExample
import math
import multiprocessing as mp
from .Constants import (
    EXPERIMENT_DESC,
    IMAGETYPE,
    IMAGESHAPE,
    SNRDB,
    NUM_NONZERO,
    NUMPROC,
    NUMTASKS,
)


def RunReconstructor_12(param):
    return ReconExample.RunReconstructor(param, [1, 2])


def RunReconstructor_pm1(param):
    return ReconExample.RunReconstructor(param, [1, -1])


if __name__ == "__main__":
    GSUP = 1 / math.sqrt(2)
    # For MAP2 reconstructor

    # For MAP1
    runArgsMap1 = [EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO]
    # For MAP2
    runArgsMap2 = [EXPERIMENT_DESC, IMAGETYPE, IMAGESHAPE, SNRDB, NUM_NONZERO, GSUP]

    fmtString = "{0}: est. hyper.={1}, perf. criteria={2}/{3}/{4}, timing={5:g}s. {6}"

    pool = mp.Pool(processes=NUMPROC)
    resultPool = pool.map(RunReconstructor_12, [runArgsMap1, runArgsMap2] * NUMTASKS)

    with open("result.txt", "w+") as fh:
        for aResult in resultPool:
            opString = fmtString.format(
                aResult["reconstructor_desc"],
                aResult["hyperparameter"],
                aResult["normalized_l2_error_norm"],
                aResult["normalized_detection_error"],
                aResult["normalized_l0_norm"],
                aResult["timing_ms"] / 1.0e3,
                aResult["termination_reason"],
            )
            fh.write(opString + "\n")
