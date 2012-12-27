import unittest
import numpy as np
import Thresholding

class T_Thresholding(unittest.TestCase):
    
    def testIdentity(self):
        t = Thresholding.ThresholdingIdentity()
        randomValues = np.random.random_sample((20,)) * 100        
        self.assertTrue(np.array_equal(randomValues, t.Apply(randomValues)), 'Thresholding id failed')
            
    def testSoft(self):
        t = Thresholding.ThresholdingSoft(1)
        randomValuesInZeroedRange = np.random.random_sample((10,)) * 2 - 1
        valuesOut = t.Apply(randomValuesInZeroedRange)
        self.assertTrue(np.equal(0, valuesOut)[0], 'Soft thresholding failed in zeroed-out range')
        randomValuesBiggerThanThreshold = 1 + np.random.random_sample((10,)) * 10
        self.assertTrue(np.array_equal(randomValuesBiggerThanThreshold - 1, t.Apply(randomValuesBiggerThanThreshold)))
        randomValuesSmallerThanThreshold = -1 - np.random.random_sample((10,)) * 10
        self.assertTrue(np.array_equal(randomValuesSmallerThanThreshold + 1, t.Apply(randomValuesSmallerThanThreshold)))
        
            
        
    
