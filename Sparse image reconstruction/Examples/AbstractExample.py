import abc

class AbstractExample(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, exampleDesc=None):
        self.exampleDesc = exampleDesc
    @abc.abstractmethod
    def RunExample(self):
        raise NotImplementedError('No default abstract method implementation')