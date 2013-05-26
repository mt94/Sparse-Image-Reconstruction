import AbstractExample

class AbstractReconstructorExample(AbstractExample.AbstractExample):
    def __init__(self, exampleDesc=None):
        super(AbstractReconstructorExample, self).__init__(exampleDesc)
        self.experimentObj = None
    