from Sim.ImageGenerator2d import RandomBinary2dImageGenerator, RandomUniform2dImageGenerator
                        
class ImageGeneratorFactory(object):
    _concreteImageGenerator = {
                               'random_binary_2d': RandomBinary2dImageGenerator,
                               'random_uniform_2d': RandomUniform2dImageGenerator
                               }    
    @staticmethod
    def GetImageGenerator(imageGeneratorDesc):
        if imageGeneratorDesc not in ImageGeneratorFactory._concreteImageGenerator:
            raise NotImplementedError("ImageGenerator " + str(imageGeneratorDesc) + " isn't implemented" )
        return ImageGeneratorFactory._concreteImageGenerator[imageGeneratorDesc]()
        
        
        
        
        
