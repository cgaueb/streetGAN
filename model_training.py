import torch
import os
import time
import shutil

import numpy as np
import pandas as pd

import lotus_dataloader
import lotus_plots
import lotus_cgan
import json
from torch.utils.data import DataLoader

class streetLightGAN :
    def __init__(self, name, device, config) :
        self.name = name
        self.config = config
        self.log_cycle = self.config['logCycle']
        self.cache_cycle = self.config['cacheCycle']
        self.validateModel = self.config['validateModel']
        self.best_score = np.inf
        self.best_uniformity = 0
        self.device = device
        self.logger = lotus_plots.trainingLog(name)
        self.model_path = os.path.join(os.getcwd(), 'meta', name)
        self.numworkers = self.config['datasetWorkers']
        self.model = None

        if not os.path.exists(self.model_path) :
            os.mkdir(self.model_path)

        self.plotter = lotus_plots.batchPlot(self.name, 'network_plots', 'visualization')

    def setDataset(self, path_to_train, path_to_test, batch_size):
        self.batch_size = batch_size
        self.train_set = lotus_dataloader.StreetDatasetGAN(path_to_train, False)
        self.test_set = lotus_dataloader.StreetDatasetGAN(path_to_test, True)

        self.trainDataloader = DataLoader(self.train_set,
            batch_size=self.batch_size, shuffle=True,
            collate_fn=None, num_workers=self.numworkers,
            drop_last=True, pin_memory=False,
            persistent_workers=True)
        
        self.testDataloader = DataLoader(self.test_set,
            batch_size=self.batch_size, shuffle=False,
            collate_fn=None, num_workers=self.numworkers,
            drop_last=True, pin_memory=False,
            persistent_workers=True)

    def buildModel(self, preTrained=False, inference=True) :
        self.train_plotter = lotus_plots.batchPlot(self.name, 'train_plots', False if preTrained else True, 'train_epoch')
        self.test_plotter = lotus_plots.batchPlot(self.name, 'train_plots', False if preTrained else True, 'test_epoch')

        self.model = lotus_cgan.streetGAN(self.plotter, self.config)
        self.model = self.model.to(device=self.device)
        self.epoch_i = 1
        modelParams = self.model.numParams()

        print(f'Model trainable params: Generator/{modelParams[0]} - Discriminator/{modelParams[1]}')

        if preTrained :
            checkpoint = torch.load(os.path.join(self.model_path, self.name))

            self.epoch_i = checkpoint['epoch'] + 1
            self.logger = checkpoint['logger']
            self.best_score = checkpoint['score']

            if 'config' in checkpoint :
                self.config = checkpoint['config']

            self.model.loadCheckpoint(checkpoint, inference, self.config)

    def train(self, num_epochs) :
        self.epochs = num_epochs + self.epoch_i
        log_path = os.path.join(os.getcwd(), 'plots')
        if not os.path.exists(log_path) :
            os.mkdir(log_path)

        for epoch_i in range(self.epoch_i, self.epochs) :
            print(f'Epoch {epoch_i}/{self.epochs - 1}')

            train_losses = self.trainStep(epoch_i)

            self.logger.epochLog('GAN_train_local/D', train_losses['lossD_local'])
            self.logger.epochLog('GAN_train_local/G', train_losses['lossG_local'])

            self.logger.epochLog('GAN_discr_local/real', train_losses['lossD_real_local'])
            self.logger.epochLog('GAN_discr_local/fake', train_losses['lossD_fake_local'])

            self.logger.epochLog('Goal_l1/train', train_losses['goal_l1_err'])
            self.logger.epochLog('l2/train', train_losses['l2_err'])
            self.logger.epochLog('Goal_uniformity/train', train_losses['goal_uniformity'])
            self.logger.epochLog('Goal_uniformity_attain/train', train_losses['goal_uniformity_attain'])

            if self.validateModel :
                valid_losses = self.validationStep(epoch_i)
                self.logger.epochLog('Goal_l1/valid', valid_losses['goal_l1_err'])
                self.logger.epochLog('Goal_uniformity/valid', valid_losses['goal_uniformity'])
                self.logger.epochLog('Goal_uniformity_attain/valid', valid_losses['goal_uniformity_attain'])

            self.logger.flush()

            if epoch_i % self.log_cycle == 0 :
                self.plotSamples(epoch_i)

            if epoch_i % self.cache_cycle == 0 :
                cache_path = os.path.join(os.getcwd(),
                    'plots', self.name, 'checkpoints')
                
                if not os.path.exists(cache_path) :
                    os.mkdir(cache_path)

                cache_path = os.path.join(cache_path, f'checkpoint_epoch_{epoch_i}')
                
                if os.path.exists(cache_path) :
                    shutil.rmtree(cache_path)

                os.mkdir(cache_path)
                self.cacheModel(epoch_i, os.path.join(cache_path, self.name), train_losses['goal_l1_err'])

            val = valid_losses['goal_l1_err'] if self.validateModel else train_losses['goal_l1_err']
            
            if val < self.best_score :
                print(f'Saving best model with mean goal: {val:.5f}')
                self.best_score = val
                self.cacheModel(epoch_i, os.path.join(self.model_path, self.name + '_cp'), val)
                                            
        self.cacheModel(self.epochs - 1, os.path.join(self.model_path, self.name + '_lastEpoch'), val)

    def trainStep(self, epoch_i) :
        self.model.train(True)
        lossesD, lossesG = None, None
        total_time = 0.
        num_batches = len(self.trainDataloader)
        discr_training_cycle = self.model.discr_training_cycle

        def appendLog(log1, log2) :
            for key in log1.keys() :
                log1[key] += (log2[key] / num_batches)
            return log1

        for batch_i, batch in enumerate(self.trainDataloader):
            t_start = time.time()
            shouldTrainGen = (batch_i + 1) % discr_training_cycle == 0 or batch_i == num_batches - 1
            dict_lossD, dict_lossG = self.model.optimizationStep(self.prepareBatch(batch), shouldTrainGen, batch_i + 1, epoch_i)

            if lossesD is not None :
                lossesD = appendLog(lossesD, dict_lossD)
            else :
                lossesD = { k : v / num_batches for k, v in dict_lossD.items() }

            if lossesG is not None :
                if shouldTrainGen :
                    lossesG = appendLog(lossesG, dict_lossG)
            else :
                if shouldTrainGen:
                    lossesG = { k : v / num_batches for k, v in dict_lossG.items() }

            t_end = time.time()
            total_time += t_end - t_start

            print(f'\tTraining batch {batch_i + 1}/{num_batches}, ', flush=False, end='')
            print(f'Elapsed time: {total_time:.2f}, ',  flush=False, end='')
            if shouldTrainGen :
                print('Goal MAE: {0:.5f}, '.format(lossesG['goal_l1_err']), flush=False, end='')
                print('Goal Uniformity: {0:.5f}, '.format(lossesG['goal_uniformity_attain']), flush=False, end='')
            
            print('', end='\r')
        print('')

        return {**lossesD, **lossesG}

    def validationStep(self, epoch_i) :
        self.model.train(False)
        metricsG = None
        total_time = 0.
        num_batches = len(self.testDataloader)

        def appendLog(log1, log2) :
            for key in log1.keys() :
                log1[key] += (log2[key] / num_batches)
            return log1

        for batch_i, batch in enumerate(self.testDataloader):
            t_start = time.time()
            metrics = self.model.validationStep(self.prepareBatch(batch), batch_i + 1, epoch_i, 8)

            if metricsG is not None :
                metricsG = appendLog(metricsG, metrics)
            else :
                metricsG = { k : v / num_batches for k, v in metrics.items() }

            t_end = time.time()
            total_time += t_end - t_start

            print(f'\tTesting batch {batch_i + 1}/{num_batches}, ', flush=False, end='')
            print(f'Elapsed time: {total_time:.2f}, ',  flush=False, end='')
            print('Goal MAE: {0:.5f}, '.format(metricsG['goal_l1_err']), flush=False, end='')
            print('Goal Uniformity: {0:.5f}, '.format(metricsG['goal_uniformity_attain']), flush=False, end='')
            
            print('', end='\r')
        print('')

        return metricsG

    def prepareBatch(self, batch) :
        rest = batch[2:]
        return [torch.permute(batch[1], (0, 3, 2, 1)).to(device=self.device, dtype=torch.int32),] +\
            [torch.permute(x, (0, 3, 2, 1)).to(device=self.device, dtype=torch.float32) for x in rest]
    
    def cacheModel(self, epoch_i, path, best_score) :
        modelDict = self.model.getDict()
        torch.save({'epoch' : epoch_i,
            'score' : best_score,
            'logger' : self.logger,
            'config' : self.config, **modelDict }, path)
        
    def plotSamples(self, epoch_i) :
        self.model.train(False)
        numSamples = 4

        def plotBatch(dataloader, plotter) :
            with torch.no_grad() :
                batch = next(iter(dataloader))
                onehot_types, goal, albedo, seg, real_gi, real_direct, real_placement, vmask = self.prepareBatch(batch)
                fake_blobs, fake_direct = self.model.generate(onehot_types, goal, seg, albedo, vmask)

                samples = torch.cat([fake_direct, fake_blobs], dim=1)

                for sample_i in range(numSamples - 1) :
                    fake_blobs, fake_direct = self.model.generate(onehot_types, goal, seg, albedo, vmask)
                    samples = torch.cat([samples, fake_direct, fake_blobs], dim=1)

                samples = torch.cat([samples, real_direct, seg[:, 2:3, ...], goal[:, 0:1, ...]], dim=1)

                plotter.plotTrainingBatchN(batch[0], samples.cpu(), f'epoch_{epoch_i}')
                
        plotBatch(self.trainDataloader, self.train_plotter)

        if self.validateModel :
            plotBatch(self.testDataloader, self.test_plotter)

    def getInfo(self) :
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def plotGenerator(self, path_to_dataset, numBatches) :
        self.buildModel(True, True)
        #self.getInfo()

        test_dataloader = DataLoader(
            lotus_dataloader.StreetDatasetGAN(path_to_dataset, True),
            batch_size=1, shuffle=False,
            num_workers=1, collate_fn=None,
            pin_memory=False, drop_last=False)
        
        self.model.train(False)

        plotter = lotus_plots.batchPlot(self.name, 'generator_plots')

        with torch.no_grad() :
            for batch_i, batch in enumerate(test_dataloader) :
                print(f'Predicing batch {batch_i + 1}/{len(test_dataloader) if numBatches < 0 else numBatches}')
                onehot_types, goal, albedo, seg, real_gi, real_direct, real_placement, vmask = self.prepareBatch(batch)
                fake_direct, fake_placement = self.model.generateN(onehot_types, goal, seg, albedo, vmask, 8)
                
                goal_pred = []

                for fake_direct_i in fake_direct :
                    goal_pred += [ self.model.eval_goal_err(fake_direct_i, real_direct, goal, seg), ]

                goal_pred += [self.model.eval_goal_err(real_direct, real_direct, goal, seg), ]

                plotter.plotGenerator(batch[0], goal_pred, fake_placement, real_placement, fake_direct, real_direct, goal[:, 0:1, ...])
                if batch_i == numBatches :
                    break

    def plotPostprocess(self, path_to_dataset, numBatches) :
        self.buildModel(True, True)

        test_dataloader = DataLoader(
            lotus_dataloader.StreetDatasetGAN(path_to_dataset, True),
            batch_size=1, shuffle=False,
            num_workers=1, collate_fn=None,
            pin_memory=False, drop_last=False)
        
        self.model.train(False)

        plotter = lotus_plots.batchPlot(self.name, 'postprocess_plots')
        path = os.path.join(os.getcwd(), 'addons', 'precomputedDL.npz')
        precombutedDL = torch.tensor(np.load(path)['a']).unsqueeze(0).permute(0, 3, 2, 1)
        precombutedDL = precombutedDL.to(self.device)

        kernel = lotus_cgan.buildKernel(11)
        kernel = kernel.expand(1, 128**2, -1)
        kernel = kernel.to(self.device)
        
        numSeeds = 8

        for batch_i, batch in enumerate(test_dataloader) :
            print(f'Predicing batch {batch_i + 1}/{len(test_dataloader) if numBatches < 0 else numBatches}')

            onehot_types, goal, albedo, seg, real_gi, real_direct, real_blobs, vmask = self.prepareBatch(batch)
            
            gen_goal_err = []
            clustered_goal_err = []
            snapped_goal_err = []
            gen_direct_list = []
            clustered_direct_list = []
            snapped_direct_list = []
            gen_blobs_list = []
            clustered_blobs_list = []
            snapped_blobs_list = []

            for seed_i in range(numSeeds) :
                gen_lightGrid, gen_blobs, clusteredGrid, clustered_blobs, snapped_placement = \
                    self.model.generateSnappedLights(onehot_types, goal, seg, albedo, vmask, precombutedDL, kernel)

                gen_direct = torch.sum(gen_lightGrid.view(1, -1, 1, 1) * precombutedDL * vmask, dim=1, keepdim=True) * albedo
                clustered_direct = torch.sum(clusteredGrid.view(1, -1, 1, 1) * precombutedDL * vmask, dim=1, keepdim=True) * albedo

                #gen_direct /= 20.
                #clustered_direct /= 20.
                
                #gen_blobs /= 20.
                #clustered_blobs /= 20.

                gen_goal_err += [ self.model.eval_goal_uniformity(gen_direct, real_direct, goal, seg, onehot_types), ]
                clustered_goal_err += [self.model.eval_goal_uniformity(clustered_direct, real_direct, goal, seg, onehot_types),]
                
                snapped_direct, snapped_blobs, _, _ = self.model.computeDLfromPlacement(snapped_placement, albedo, seg, goal, onehot_types, kernel)

                #snapped_direct /= 20.
                #snapped_blobs /= 20.

                gen_direct_list += [gen_direct,]
                clustered_direct_list += [clustered_direct,]

                gen_blobs_list += [gen_blobs,]
                clustered_blobs_list += [clustered_blobs,]

                snapped_direct_list += [snapped_direct,]
                snapped_blobs_list += [snapped_blobs,]

                snapped_goal_err += [self.model.eval_goal_uniformity(snapped_direct, real_direct, goal, seg, onehot_types),]

            real_goal_err = self.model.eval_goal_uniformity(real_direct, real_direct, goal, seg, onehot_types)

            plotter.plotPostProcess(batch[0],
                gen_goal_err, clustered_goal_err, snapped_goal_err, real_goal_err,
                gen_blobs_list, clustered_blobs_list, snapped_blobs_list, real_blobs,
                gen_direct_list, clustered_direct_list, snapped_direct_list, real_direct, 
                goal[:, 0:1, ...], seg)

            if (batch_i + 1) == numBatches :
                break

    def evaluate(self, path_to_dataset, applyPostProc) :
        self.buildModel(True, True)

        test_dataloader = DataLoader(
            lotus_dataloader.StreetDatasetGAN(path_to_dataset, True),
            batch_size=1, shuffle=False,
            num_workers=1, collate_fn=None,
            pin_memory=False, drop_last=False)
        
        self.model.train(False)

        out_path = os.path.join(os.getcwd(), 'plots', self.name, 'evaluation')
        if not os.path.exists(out_path) :
            os.mkdir(out_path)

        numSamples = 4

        def buildLog() :
            dict_log = { f'fake_{i + 1 }' : [] for i in range(numSamples) }
            dict_log['real'] = []
            dict_log['mean_err_fake'] = []
            return dict_log

        dict_logs = {
            'log_u' : buildLog(),
            'log_l' : buildLog(),
            'log_u_per' : buildLog(),
            'log_err' : buildLog(),
            'log_rel_err' : buildLog(),
            'log_px_succ_per' : buildLog(),
            'log_px_below_per' : buildLog(),
            }

        for batch_i, batch in enumerate(test_dataloader) :
            print(f'Predicing batch {batch_i + 1}/{len(test_dataloader)}')
            onehot_types, goal, albedo, seg, real_gi, real_direct, real_placement, vmask = self.prepareBatch(batch)
            fake_direct_list, _ = self.model.generateN(onehot_types, goal, seg, albedo, vmask, numSamples, optimized=applyPostProc)
            
            real_u_per, real_u, real_l, real_avg_err, real_err, real_rel_err = self.model.eval_goal_uniformity(real_direct, real_direct, goal, seg, onehot_types)
            real_succ, real_below = self.model.eval_goal_attain(real_direct, goal, seg)
            
            dict_logs['log_u']['real'] += [round(real_u, 5)]
            dict_logs['log_l']['real'] += [round(real_l, 5)]
            dict_logs['log_u_per']['real'] += [round(real_u_per, 5)]
            dict_logs['log_err']['real'] += [round(real_err, 5)]
            dict_logs['log_rel_err']['real'] += [round(real_rel_err, 5)]
            dict_logs['log_px_succ_per']['real'] += [round(real_succ, 5)]
            dict_logs['log_px_below_per']['real'] += [round(real_below, 5)]
            
            for i, fake_direct_i in enumerate(fake_direct_list) :
                fake_u_per, fake_u, fake_l, fake_avg_err, fake_err, fake_rel_err = self.model.eval_goal_uniformity(fake_direct_ if applyPostProc else fake_direct_i, real_direct, goal, seg, onehot_types)
                fake_succ, fake_below = self.model.eval_goal_attain(fake_direct_i if applyPostProc else fake_direct_i, goal, seg)
                dict_logs['log_u'][f'fake_{i + 1}'] += [round(fake_u, 5)]
                dict_logs['log_l'][f'fake_{i + 1}'] += [round(fake_l, 5)]
                dict_logs['log_u_per'][f'fake_{i + 1}'] += [round(fake_u_per, 5)]
                dict_logs['log_err'][f'fake_{i + 1}'] += [round(fake_err, 5)]
                dict_logs['log_rel_err'][f'fake_{i + 1}'] += [round(fake_rel_err, 5)]
                dict_logs['log_px_succ_per'][f'fake_{i + 1}'] += [round(fake_succ, 5)]
                dict_logs['log_px_below_per'][f'fake_{i + 1}'] += [round(fake_below, 5)]

                if i > 0 :
                    dict_logs['log_u']['mean_err_fake'][-1] += round(fake_u, 5) / numSamples
                    dict_logs['log_l']['mean_err_fake'][-1] += round(fake_l, 5) / numSamples
                    dict_logs['log_u_per']['mean_err_fake'][-1] += round(fake_u_per, 5) / numSamples
                    dict_logs['log_err']['mean_err_fake'][-1] += round(fake_err, 5) / numSamples
                    dict_logs['log_rel_err']['mean_err_fake'][-1] += round(fake_rel_err, 5) / numSamples
                    dict_logs['log_px_succ_per']['mean_err_fake'][-1] += round(fake_succ, 5) / numSamples
                    dict_logs['log_px_below_per']['mean_err_fake'][-1] += round(fake_below, 5) / numSamples
                else :
                    dict_logs['log_u']['mean_err_fake'] += [round(fake_u, 5) / numSamples]
                    dict_logs['log_l']['mean_err_fake'] += [round(fake_l, 5) / numSamples]
                    dict_logs['log_u_per']['mean_err_fake'] += [round(fake_u_per, 5) / numSamples]
                    dict_logs['log_err']['mean_err_fake'] += [round(fake_err, 5) / numSamples]
                    dict_logs['log_rel_err']['mean_err_fake'] += [round(fake_rel_err, 5) / numSamples]
                    dict_logs['log_px_succ_per']['mean_err_fake'] += [round(fake_succ, 5) / numSamples]
                    dict_logs['log_px_below_per']['mean_err_fake'] += [round(fake_below, 5) / numSamples]

        for key, value in dict_logs.items() :
            df_log = pd.DataFrame(value)
            df_log.to_csv(os.path.join(out_path, '{0}_{1}.csv'.format(key, 'opt' if applyPostProc else 'noopt')))

    def exportPredictedLights(self, path_to_dataset) :
        self.buildModel(True, True)

        test_dataloader = DataLoader(
            lotus_dataloader.StreetDatasetGAN(path_to_dataset, True),
            batch_size=1, shuffle=False, num_workers=1, collate_fn=None,
            pin_memory=False, drop_last=False)
        
        self.model.train(False)
        numGenerations = 8
        jsonPath = os.path.join(os.getcwd(), 'plots', self.name)
        jsonData = {}
        blocks = []

        for batch_i, batch in enumerate(test_dataloader) :
            names = batch[0]

            print(f'processing {batch_i+1}/{len(test_dataloader)} - street id {names}')

            onehot_types, goal, albedo, seg, real_gi, real_direct, real_placement, vmask = self.prepareBatch(batch)

            block = {}
            block['street_id'] = int(names[0])
            block['light_gen'] = []

            for gen_i in range(numGenerations) :
                light_placement = self.model.generateOptLights(goal, seg, albedo, onehot_types)[0].cpu().permute(2, 1, 0)
                light_pos = []

                light_indices = np.where(light_placement > 0)
                light_positions = np.array(list(zip(light_indices[0].ravel(), light_indices[1].ravel())), dtype=np.int32)
                for placement_i in light_positions :
                    w = int(placement_i[0])
                    h = int(placement_i[1])
                    light_pos += [[w, h, light_placement[w, h, 0].item() * 2 * np.pi],]

                block['light_gen'] += [light_pos,]

            blocks += [block,]

        jsonData['blocks'] = blocks
        with open(os.path.join(jsonPath, 'predicted_lights.json'), 'w') as outfile :
            json.dump(jsonData, outfile, indent=4)