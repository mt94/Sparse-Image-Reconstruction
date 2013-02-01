import abc

class AbstractConvolutionMatrix(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        super(AbstractConvolutionMatrix, self).__init__()
        
    @abc.abstractproperty
    def PsfShape(self):
        raise NotImplementedError('No default abstract method implemented')
                    
    @abc.abstractmethod
    def Multiply(self, theta):
        raise NotImplementedError('No default abstract method implemented')
    
    @abc.abstractmethod
    def MultiplyPrime(self, theta):
        raise NotImplementedError('No default abstract method implemented')        