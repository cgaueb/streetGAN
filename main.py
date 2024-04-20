

import lotus_common
import model_training
import os
import lotus_config
from enum import Enum

def trainGANModel(name, train_path, test_path, device, batch_size, epochs, continueTraining) :
    model = model_training.streetLightGAN(name, device=device, config=lotus_config.getConfig())
    model.setDataset(train_path, test_path, batch_size=batch_size)
    model.buildModel(continueTraining, inference=False)
    model.train(num_epochs=epochs)

class TEST_MODE(Enum):
    PLOT_GENERATOR = 1
    PLOT_POSTPROCESS = 2
    EVAL_MODEL = 3
    EXPORT_LIGHTS = 4

def testGANModel(name, test_path, device, numSamples, test_mode) :
    model = model_training.streetLightGAN(name, device=device, config=lotus_config.getConfig())

    if test_mode == TEST_MODE.PLOT_GENERATOR :
        model.plotGenerator(test_path, numSamples)
    elif test_mode == TEST_MODE.PLOT_POSTPROCESS :
        model.plotPostprocess(test_path, numSamples)
    elif test_mode == TEST_MODE.EVAL_MODEL :
        model.evaluate(test_path, False)
        model.evaluate(test_path, True)
    elif test_mode == TEST_MODE.EXPORT_LIGHTS :
        model.exportPredictedLights(test_path)

def modelTrain() :
    device = lotus_common.init_torch()
    train_path = os.path.join(os.getcwd(), 'dataset', 'train', '*.npz')
    test_path = os.path.join(os.getcwd(), 'dataset', 'test', '*.npz')

    lotus_config.useGenBaseline = False
    lotus_config.useGeneratorL1Loss = True
    lotus_config.generatorL1Mult = 10
    lotus_config.useDiscriminatorRelError = True
    lotus_config.discriminatorErrThresh = 0.2
    lotus_config.validateModel = True
    lotus_config.cacheCycle = 5
    trainGANModel('model_streetGan', train_path, test_path, device, 8, 100, False)

def modelTest() :
    device = lotus_common.init_torch()
    train_path = os.path.join(os.getcwd(), 'dataset', 'train', '*.npz')
    test_path = os.path.join(os.getcwd(), 'dataset', 'test', '*.npz')

    testGANModel('model_streetGan', test_path, device, -1, TEST_MODE.EVAL_MODEL)
    testGANModel('model_streetGan', test_path, device, -1, TEST_MODE.EXPORT_LIGHTS)
    testGANModel('model_streetGan', test_path, device, -1, TEST_MODE.PLOT_POSTPROCESS)
    testGANModel('model_streetGan', test_path, device, -1, TEST_MODE.PLOT_GENERATOR)

def main() :
    modelTrain()
    #modelTest()

if __name__ == '__main__':
    main()