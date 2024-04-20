useGeneratorL1Loss = False
generatorL1Mult = 1.
useGenBaseline = False

useDiscriminatorRelError = True
discriminatorErrThresh = 0.2

genLearningRate = 0.0001
discrLearningRate = genLearningRate * 2
useAMSgrad = True
adamBeta1 = 0.
adamBeta2 = 0.999

noiseDim = 256
generatorInitFilters = 64
discriminatorInitFilters = 32
discriminatorTrainingCycle = 1
discriminatorNormFactor = 20 #normalises the input image illuminance, see paper for details.

validateModel = False
cacheCycle = 5
logCycle = 1
datasetWorkers = 6#multiprocessing.cpu_count() - 4

def getConfig() :
    return { \
        'useGeneratorL1Loss' : useGeneratorL1Loss,
        'generatorL1Mult' : generatorL1Mult,
        'useDiscriminatorRelError' : useDiscriminatorRelError,
        'discriminatorErrThresh' : discriminatorErrThresh,
        'genLearningRate' : genLearningRate,
        'discrLearningRate' : discrLearningRate,
        'noiseDim' : noiseDim,
        'generatorInitFilters' : generatorInitFilters,
        'discriminatorInitFilters' : discriminatorInitFilters,
        'validateModel' : validateModel,
        'cacheCycle' : cacheCycle,
        'logCycle' : logCycle,
        'datasetWorkers' : datasetWorkers,
        'discriminatorTrainingCycle' : discriminatorTrainingCycle,
        'discriminatorNormFactor' : discriminatorNormFactor,
        'useAMSgrad' : useAMSgrad,
        'adamBeta1' : adamBeta1,
        'adamBeta2' : adamBeta2,
        'useGenBaseline' : useGenBaseline
    }
