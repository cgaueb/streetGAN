import os
import shutil
import pandas as pd
import numpy as np
import cv2 as cv

import torch

import matplotlib.pyplot as plt
import seaborn as sns

class trainingLog() :
    def __init__(self, outPath) :
        self.path = os.path.join(os.getcwd(), 'plots', outPath)

        if not os.path.exists(self.path) :
            os.mkdir(self.path)
            
        self.scalar_log = dict()

    def epochLog(self, tag, y):
        self.__addLog(self.scalar_log, tag.split('/'), y)

    def flush(self):
        self.fig = plt.figure('log')
        for log_tag, log_values in self.scalar_log.items() :
            log_array = np.zeros(shape=(
                len(log_values[list(log_values.keys())[0]]),
                len(log_values)), dtype=np.float32)

            for i, (_, plot_values) in enumerate(log_values.items()) :
                log_array[:, i] = np.array(plot_values)

            df_log = pd.DataFrame(log_array, columns=list(log_values.keys()))

            self.fig.clf()
            ax = plt.axes()
            ax.grid(True)
            
            for column in log_values.keys() :
                sns.lineplot(ax=ax, data=df_log, x=df_log.index, y=column)

            ax.legend(labels=list(log_values.keys()))

            self.fig.savefig(os.path.join(self.path, log_tag + '.png'), bbox_inches='tight')
            df_log.to_csv(os.path.join(self.path, log_tag + '.csv'))

        plt.close(self.fig)

    def __addLog(self, logDict, tokens, y) :
        token = tokens[0]
        if token in logDict :
            if len(tokens) > 1 :
                self.__addLog(logDict[token], tokens[1:], y)
            else :
                logDict[token] += [y,]
        else :
            if len(tokens) > 1 :
                tmp = dict()
                self.__addLog(tmp, tokens[1:], y)
                logDict[token] = tmp
            else :
                logDict[token] = [y,]

class batchPlot() :
    def __init__(self, modelName, rootFolderName, clearPath=True, folderName=None) :
        self.rootFolderName = rootFolderName
        self.name = folderName
        self.path = os.path.join(os.getcwd(), 'plots', modelName, rootFolderName)

        if not os.path.exists(self.path) :
            os.mkdir(self.path)

        if self.name != None :
            self.path = os.path.join(self.path, self.name)

            if clearPath :
                if os.path.exists(self.path) :
                    shutil.rmtree(self.path)

                os.mkdir(self.path)

        self.fig = plt.figure('plots')

    def tonemap(self, x, rescale=True) :
        norm_factor = 20. if rescale else 1.
        x = np.log(norm_factor * x + 1)
        a = 0.055
        return np.where(x <= 0.0031308, 12.92*x, (1 + a) * np.power(x, 1/2.4) - a)
        
    def plotGANBatch(self, x, y, numSamples, suffix, folderName) :
        outPath = os.path.join(self.path, folderName)

        if not os.path.exists(outPath) :
            os.mkdir(outPath)
        
        x = x[:numSamples, ...].clone().detach()
        y = y[:numSamples, ...].clone().detach()
        pred_stacked = torch.stack([x, y], dim=1).cpu()
        
        for sample_i, sample in enumerate(pred_stacked) :
            x = sample[0, ::].permute(2, 1, 0)
            y = sample[1, ::].permute(2, 1, 0)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title('Generated')
            ax1.imshow(self.tonemap(x), cmap='gray')
            ax2.set_title('Reference')
            ax2.imshow(self.tonemap(y), cmap='gray')
            fig.savefig(os.path.join(outPath, f'{sample_i}_{suffix}.png'))
            plt.close(fig)

    def plotPostProcess(self, sampleName,
        gen_goalErr, clustered_goalErr, snapped_goalErr, real_goalErr,
        gen_blobs_list, clustered_blobs_list, snapped_blobs_list, real_blobs,
        gen_direct_list, clustered_direct_list, snapped_direct_list, real_direct,
        goal, seg) :

        def plotAxis(axisList, samples, goals) :
            for ax_i, ax in enumerate(axisList) :
                if ax_i == len(axisList) - 3 :
                    ax.set_title(f'Ref-{real_goalErr[0]:.4f} / {real_goalErr[4]:.4f}')
                elif ax_i == len(axisList) - 2 :
                    ax.set_title('goal')
                elif ax_i == len(axisList) - 1 :
                    ax.set_title('mask')
                else :
                    ax.set_title(f'{goals[ax_i][0]:.4f} / {goals[ax_i][4]:.4f}')
                    
                if ax_i == len(axisList) - 1 :
                    ax.imshow(seg[0].cpu().permute(2, 1, 0))
                else :
                    img = samples[ax_i, ::].permute(2, 1, 0)
                    ax.imshow(self.tonemap(img), cmap='gray')

        fig, axisList = plt.subplots(6, len(gen_blobs_list) + 2, figsize=(30, 20))

        plotAxis(axisList[0, :], torch.stack(gen_direct_list + [real_direct, goal], dim=1).cpu()[0], gen_goalErr)
        plotAxis(axisList[1, :], torch.stack(gen_blobs_list + [real_blobs, goal], dim=1).cpu()[0], gen_goalErr)

        plotAxis(axisList[2, :], torch.stack(clustered_direct_list + [real_direct, goal], dim=1).cpu()[0], clustered_goalErr)
        plotAxis(axisList[3, :], torch.stack(clustered_blobs_list + [real_blobs, goal], dim=1).cpu()[0], clustered_goalErr)

        plotAxis(axisList[4, :], torch.stack(snapped_direct_list + [real_direct, goal], dim=1).cpu()[0], snapped_goalErr)
        plotAxis(axisList[5, :], torch.stack(snapped_blobs_list + [real_blobs, goal], dim=1).cpu()[0], snapped_goalErr)

        fig.savefig(os.path.join(self.path, f'{sampleName[0]}.png'))
        plt.close(fig)
        
    def plotGenerator(self, sampleNames, goalErr, fake_placement_list, real_placement, fake_direct_list, real_direct, goal) :
        pred_stacked = torch.stack(
            fake_direct_list + [real_direct, goal,] + 
            fake_placement_list + [real_placement, goal,], dim=1).cpu()
        
        def plotAxis(axisList, samples) :
            for ax_i, ax in enumerate(axisList) :
                if ax_i == len(axisList) - 2 :
                    ax.set_title(f'Ref-{goalErr[ax_i][0]:.4f} / {goalErr[ax_i][1]:.4f}')
                elif ax_i == len(axisList) - 1 :
                    ax.set_title('goal')
                else :
                    ax.set_title(f'{goalErr[ax_i][0]:.4f} / {goalErr[ax_i][1]:.4f}')
                    
                img = samples[ax_i, ::].permute(2, 1, 0)
                ax.imshow(self.tonemap(img), cmap='gray')

        for sample_i, sample in enumerate(pred_stacked) :
            numRowPlots = int(sample.size(0) / 2)
            fig, axisList = plt.subplots(2, numRowPlots, figsize=(30, 5))

            axisList_direct = axisList[0, :]
            axisList_placement = axisList[1, :]
            samples_direct = sample[:numRowPlots, ::]
            samples_gi = sample[numRowPlots:, ::]
            plotAxis(axisList_direct, samples_direct)
            plotAxis(axisList_placement, samples_gi)
            fig.savefig(os.path.join(self.path, f'{sampleNames[sample_i]}.png'))
            #plt.show()
            plt.close(fig)

    def plotTrainingBatch(self, sampleNames, fake_blobs, fake_direct, real_gi, real_direct, real_placement, seg, goal, folderName) :
        outPath = os.path.join(self.path, folderName)

        if not os.path.exists(outPath) :
            os.mkdir(outPath)

        pred_stacked = torch.stack([fake_blobs, fake_direct, real_direct, real_placement, seg, goal[:, 0:1, ...], goal[:, 1:2, ...]], dim=1).cpu()

        for sample_i, sample in enumerate(pred_stacked) :
            fake_blobs_i = sample[0, ::].permute(2, 1, 0)
            fake_direct_i = sample[1, ::].permute(2, 1, 0)
            real_direct = sample[2, ::].permute(2, 1, 0)
            real_blobs = sample[3, ::].permute(2, 1, 0)
            seg_i = sample[4, ::].permute(2, 1, 0)
            meanGoal_i = sample[5, ::].permute(2, 1, 0)

            fig, (ax1, ax2, ax3, ax4, ax5, ax6,) = plt.subplots(1, 6, figsize=(4 * 6, 4))

            ax1.set_title('fake - direct')
            ax1.imshow(self.tonemap(fake_direct_i, True), cmap='gray')

            ax2.set_title('fake - blobs')
            ax2.imshow(self.tonemap(fake_blobs_i, True), cmap='gray')

            ax3.set_title('real - direct')
            ax3.imshow(self.tonemap(real_direct, True), cmap='gray')

            ax4.set_title('real - blobs')
            ax4.imshow(self.tonemap(real_blobs, True), cmap='gray')

            ax5.set_title('real mask')
            ax5.imshow(seg_i, cmap='gray')

            ax6.set_title('real goal')
            ax6.imshow(meanGoal_i, cmap='gray')

            fig.savefig(os.path.join(outPath, sampleNames[sample_i] + '.png'))
            plt.close(fig)

    def plotTrainingBatchN(self, sampleNames, samples, folderName) :
        outPath = os.path.join(self.path, folderName)

        if not os.path.exists(outPath) :
            os.mkdir(outPath)

        for sample_i, sample in enumerate(samples) :
            fig, axisList = plt.subplots(1, samples.size(1), figsize=(samples.size(1) * 5, 5))

            for ax_i, ax in enumerate(axisList) :
                if ax_i == len(axisList) - 1 :
                    ax.set_title('goal')
                elif ax_i == len(axisList) - 2 :
                    ax.set_title(f'road')
                elif ax_i == len(axisList) - 3 :
                    ax.set_title(f'ref')
                else :
                    ax.set_title(f'seed')
                    
                img = sample[ax_i, ::].permute(1, 0)
                ax.imshow(self.tonemap(img), cmap='gray')

            fig.savefig(os.path.join(outPath, sampleNames[sample_i] + '.png'))
            plt.close(fig)