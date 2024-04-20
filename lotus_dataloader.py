import numpy as np
import os
import torch
import glob

class StreetDatasetGAN(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset, testSet=False):
        
        self.dataset_folder = path_to_dataset
        self.folders = glob.glob(path_to_dataset)
        self.testSet = testSet
        self.mipmap_max = 7 #2^7
        self.mipmap_min = 3 #2^3
        self.numSampleFolders = -1 if testSet else 1000

    def __len__(self):
        return self.numSampleFolders if self.numSampleFolders > 0 else len(self.folders)

    
    def __getitem__(self, index):
        if not self.testSet:
            index = np.random.randint(0, len(self.folders))

        data = np.load(self.folders[index], mmap_mode='r')
        name = os.path.splitext(os.path.basename(self.folders[index]))[0]

        onehot_types = data['a']
        light_placement = data['b']
        ref = data['c']
        goal = data['d']
        ref_direct = data['e']
        vmask = data['f']
        objcodes = data['h']

        light_placement = np.mean(light_placement, axis=-1, keepdims=True)
        ref = np.mean(ref, axis=-1, keepdims=True)
        ref_direct = np.mean(ref_direct, axis=-1, keepdims=True)

        seg = np.zeros(shape=(onehot_types.shape[0], onehot_types.shape[1], 3), dtype=np.float32)
        seg[:, :, 0] = onehot_types[:, :, 0] #background
        seg[:, :, 1] = onehot_types[:, :, 1] #buildings
        seg[np.logical_and(seg[:, :, 0] == 0, seg[:, :, 1] == 0), 2] = 1 #roads

        ref[seg[:, :, 1] == 1] = 0
        ref_direct[seg[:, :, 1] == 1] = 0
        light_placement[seg[:, :, 1] == 1] = 0
        
        #input is over spherical point lights (/4*pi), convert to hemispherical
        ref *= 2.
        ref_direct *= 2.
        light_placement *= 2.
        goal *= 2.

        goal = np.repeat(goal, 2, axis=-1)
        
        albedo = np.zeros_like(ref)
        albedo[seg[:, :, 0] == 1] = 1
        albedo[seg[:, :, 2] == 1] = 0.7

        return name, objcodes, goal, albedo, seg, ref, ref_direct, light_placement, vmask