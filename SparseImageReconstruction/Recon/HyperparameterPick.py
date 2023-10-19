import numpy as np
import pylab as plt
from .AbstractIterationsObserver import AbstractIterationsObserver

class HyperparameterPick(object):
    def __init__(self, iterObserver):
        assert isinstance(iterObserver, AbstractIterationsObserver)
        assert (len(iterObserver.HistoryState) > 0) and (len(iterObserver.HistoryState) == len(iterObserver.HistoryEstimate))
        self._iterObserver = iterObserver
            
    @staticmethod
    def _indexMin(values):
        return min(xrange(len(values)),key=values.__getitem__)
            
    def _GetMetric(self, metricDesc):
        try:
            metricVec = []
            for h in self._iterObserver.HistoryState:
                metricVec.append(h[metricDesc])
            return metricVec
        except KeyError:
            return None          
              
    def PlotMetricVsStages(self, metricDesc, fignum):
        """ Plots the specified metric vs. the number of stages """
        if fignum < 1:
            raise ValueError('fignum must be strictly positive')
        metricVec = self._GetMetric(metricDesc)
        if (metricVec is not None) and (len(metricVec) > 0):
            plt.figure(fignum)
            plt.plot(np.arange(len(metricVec)) + 1, metricVec, 'bs-')
            plt.grid()
            plt.xlabel('LARS iteration')
            plt.ylabel('SURE criterion')
                        
    def GetBestEstimate(self, metricDesc, metricDeltaThres=None):
        """ 
        Get the best estimate, where 'best' is determined by looking at metricDesc. The 'best'
        may not necessarily be the minimum, since the metric may constantly decrease and then
        'saturate'. In this case, we're interested in the point at which further decreases are
        insignificant.
        """
        metricVec = self._GetMetric(metricDesc)
        
        if metricDeltaThres is not None:
            assert metricDeltaThres < 0, 'Metric delta threshold must be strictly negative'
            metricDeltaVec = np.diff(np.array(metricVec))
            indSmallDelta = np.where(metricDeltaVec > metricDeltaThres)[0]            
            if indSmallDelta.size > 0:
                indBest = indSmallDelta[0]
                return (indBest, self._iterObserver.HistoryEstimate[indBest])
            else:
                # If none of the elements are larger than metricDeltaThres (a negative number),
                # return the last estimate                
                return (len(metricVec) - 1, self._iterObserver.HistoryEstimate[-1])
        else:
            # Fallback case: find the min metric and return the estimate for that iteration
            metricMinInd = self._indexMin(metricVec)
            return (metricMinInd, self._iterObserver.HistoryEstimate[metricMinInd])
