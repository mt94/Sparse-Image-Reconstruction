from ..Sim.ImageGeneratorImpl import (
    SparseBinaryImageGenerator,
    SparseDiscreteImageGenerator,
    SparseUniformImageGenerator,
)


class ImageGeneratorFactory(object):
    _concreteImageGenerator = {
        "random_binary": SparseBinaryImageGenerator,
        "random_discrete": SparseDiscreteImageGenerator,
        "random_uniform": SparseUniformImageGenerator,
    }

    @staticmethod
    def GetImageGenerator(imageGeneratorDesc):
        if imageGeneratorDesc not in ImageGeneratorFactory._concreteImageGenerator:
            raise NotImplementedError(
                "ImageGenerator " + str(imageGeneratorDesc) + " isn't implemented"
            )
        return ImageGeneratorFactory._concreteImageGenerator[imageGeneratorDesc]()
