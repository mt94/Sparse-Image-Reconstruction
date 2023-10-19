import pylab as plt
from .AbstractExample import AbstractExample
from ..Systems.NumericalHelper import NumericalHelper


class TruncatedStandardNormalVariates(AbstractExample):
    def __init__(self):
        super(TruncatedStandardNormalVariates, self).__init__(
            "Generated truncated standard normal variates"
        )

    def RunExample(self):
        numSamples = 1000
        self.GenerateNoTruncation(numSamples, 1)
        self.GenerateTruncationUsingSmallLimits(numSamples, 2)
        self.GenerateNonpositive(numSamples, 3)
        self.GenerateTruncationUsingLargeLimitsOneSided(numSamples, 4)
        self.GenerateTruncationUsingLargeLimitsTwoSided(numSamples, 5)

    _layout = 120
    _histNumBins = 20

    @staticmethod
    def GenerateNoTruncation(numSamples, fignum):
        rv = NumericalHelper.RandomTruncatedStandardNormal(n=numSamples)
        plt.figure(fignum)
        plt.subplot(TruncatedStandardNormalVariates._layout + 1)
        plt.plot(rv, "g.")
        plt.ylabel("No truncation")
        plt.subplot(TruncatedStandardNormalVariates._layout + 2)
        plt.hist(rv, TruncatedStandardNormalVariates._histNumBins)

    @staticmethod
    def GenerateTruncationUsingSmallLimits(numSamples, fignum):
        rv = NumericalHelper.RandomTruncatedStandardNormal(-2, 3, numSamples)
        plt.figure(fignum)
        plt.subplot(TruncatedStandardNormalVariates._layout + 1)
        plt.plot(rv, "b.")
        plt.ylabel("Truncation using a=-2, b=-3")
        plt.subplot(TruncatedStandardNormalVariates._layout + 2)
        plt.hist(rv, TruncatedStandardNormalVariates._histNumBins)

    @staticmethod
    def GenerateNonpositive(numSamples, fignum):
        rv = NumericalHelper.RandomTruncatedStandardNormal(b=0, n=numSamples)
        plt.figure(fignum)
        plt.subplot(TruncatedStandardNormalVariates._layout + 1)
        plt.plot(rv, "m.")
        plt.ylabel("Truncation using a=-Inf, b=0")
        plt.subplot(TruncatedStandardNormalVariates._layout + 2)
        plt.hist(rv, TruncatedStandardNormalVariates._histNumBins)

    @staticmethod
    def GenerateTruncationUsingLargeLimitsOneSided(numSamples, fignum):
        rv = NumericalHelper.RandomTruncatedStandardNormal(a=40, n=numSamples)
        plt.figure(fignum)
        plt.subplot(TruncatedStandardNormalVariates._layout + 1)
        plt.plot(rv, "r.")
        plt.ylabel("Truncation using a=40, b=Inf")
        plt.subplot(TruncatedStandardNormalVariates._layout + 2)
        plt.hist(rv, TruncatedStandardNormalVariates._histNumBins)

    @staticmethod
    def GenerateTruncationUsingLargeLimitsTwoSided(numSamples, fignum):
        rv = NumericalHelper.RandomTruncatedStandardNormal(a=-1, b=40, n=numSamples)
        plt.figure(fignum)
        plt.subplot(TruncatedStandardNormalVariates._layout + 1)
        plt.plot(rv, "r.")
        plt.ylabel("Truncation using a=-1, b=40")
        plt.subplot(TruncatedStandardNormalVariates._layout + 2)
        plt.hist(rv, TruncatedStandardNormalVariates._histNumBins)


if __name__ == "__main__":
    ex = TruncatedStandardNormalVariates()
    ex.RunExample()
    plt.show()
