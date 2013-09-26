from Sim.ImageGeneratorImpl import SparseBinary2dImageGenerator, SparseBinary3dImageGenerator, SparseUniform2dImageGenerator
                        
class ImageGeneratorFactory(object):
    _concreteImageGenerator = {
                               'random_binary_2d': SparseBinary2dImageGenerator,
                               'random_binary_3d': SparseBinary3dImageGenerator,
                               'random_uniform_2d': SparseUniform2dImageGenerator
                               }    
    @staticmethod
    def GetImageGenerator(imageGeneratorDesc):
        if imageGeneratorDesc not in ImageGeneratorFactory._concreteImageGenerator:
            raise NotImplementedError("ImageGenerator " + str(imageGeneratorDesc) + " isn't implemented" )
        return ImageGeneratorFactory._concreteImageGenerator[imageGeneratorDesc]()
    