#!/c/Python27/python

from ..Examples import SimpleThresholdingReconstructorExample as ReconExample
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
    """
    Run a comparison between standard Landweber iterations and iterations
    with a non-negative thresholding operation.
    """

    runArgsLw = [
        "landweber",
        5e5,
        EXPERIMENT_DESC,
        IMAGETYPE,
        IMAGESHAPE,
        SNRDB,
        NUM_NONZERO,
    ]
    runArgsLwNneg = [
        "landweber_nonneg",
        5e5,
        EXPERIMENT_DESC,
        IMAGETYPE,
        IMAGESHAPE,
        SNRDB,
        NUM_NONZERO,
    ]

    pool = mp.Pool(processes=NUMPROC)

    resultPool = pool.map(RunReconstructor_12, [runArgsLw, runArgsLwNneg] * NUMTASKS)

    fmtString = "{0}: perf. criteria={1}/{2}/{3}, timing={4:g}s. {5}"

    with open("result.txt", "w+") as fh:
        for aResult in resultPool:
            opString = fmtString.format(
                aResult["estimator"],
                aResult["normalized_l2_error_norm"],
                aResult["normalized_detection_error"],
                aResult["normalized_l0_norm"],
                aResult["timing_ms"] / 1.0e3,
                aResult["termination_reason"],
            )
            fh.write(opString + "\n")
