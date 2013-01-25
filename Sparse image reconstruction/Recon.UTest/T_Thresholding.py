import unittest
import numpy as np
import Systems.Thresholding as thr 

class T_Thresholding(unittest.TestCase):    
    def testIdentity(self):
        t = thr.ThresholdingIdentity()
        randomValues = np.random.random_sample((20,1)) * 100        
        self.assertTrue(np.array_equal(randomValues, t.Apply(randomValues)), 'Thresholding id failed')
        
    def testSoft(self):
        t = thr.ThresholdingSoft(1)
        self.assertTrue(np.array_equal(np.array([-1, 0, 0, 1]), t.Apply(np.array([-2, -0.5, 0.5, 2]))),
                        'Soft thresholding not returning expected values')
        
    def testHard(self):
        t = thr.ThresholdingHard(1)
        self.assertTrue(np.array_equal(np.array([-2, 0, 0, 2]), t.Apply(np.array([-2, -0.5, 0.5, 2]))), 
                        'Hard thresholding not returning expected values')
    
    def testHybrid(self):
        t = thr.ThresholdingHybrid(1, 0.5)
        self.assertTrue(np.array_equal(np.array([-1.5, 0, 0, 1.5]), t.Apply(np.array([-2, -0.5, 0.5, 2]))), 
                                       'Hybrid thresholding not returning expected values')
            
        
    
