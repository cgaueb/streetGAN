import os
import glob
import shutil
import torch
import multiprocessing
import more_itertools
import math

import numpy as np
import cv2 as cv

from numba import cuda
from functools import cmp_to_key
from concurrent.futures import ProcessPoolExecutor
from numba.cuda.random import xoroshiro128p_uniform_float32

def init_torch() :
    print('Cuda enabled: {0}'.format(torch.cuda.is_available()))
    print('Cudnn enabled: {0}'.format(torch.backends.cudnn.enabled))
    #print('Test cuda array: {0}'.format(torch.zeros(1).cuda()))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.utils.backcompat.broadcast_warning.enabled=True
    #torch.autograd.set_detect_anomaly(True)

    if not os.path.exists(os.path.join(os.getcwd(), 'meta')) :
        os.mkdir(os.path.join(os.getcwd(), 'meta'))

    if not os.path.exists(os.path.join(os.getcwd(), 'addons')) :
        os.mkdir(os.path.join(os.getcwd(), 'addons'))

    if not os.path.exists(os.path.join(os.getcwd(), 'plots')) :
        os.mkdir(os.path.join(os.getcwd(), 'plots'))

    if not os.path.isfile(os.path.join(os.getcwd(), 'addons', 'precomputedDL.npz')) :
        print('Failed to locate precomputedDL. Generating DL cache...')
        genPrecomputedDL()

    return device

def genPrecomputedDL() :
    x = np.arange(start=5, stop=128 - 5, step=5)
    y = np.arange(start=5, stop=128 - 5, step=5)
    lightPlacements = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    h = 4
    accums = 10
    ref = np.zeros((128, 128, lightPlacements.shape[0]), dtype=np.float32)

    for x in range(128) :
        for y in range(128) :
            for accum_i in range(1, accums + 1) :
                a = 1. / accum_i
                _x = x + np.random.rand(1)[0]
                _y = y + np.random.rand(1)[0]
                current_pos = np.array([_x, 0, _y], dtype=np.float32)
                illum = 0

                for light_i, light in enumerate(lightPlacements) :
                    GtoL = np.array([light[0], h, light[1]], dtype=np.float32) - current_pos
                    GtoL /= np.linalg.norm(GtoL, 2)
                    d = np.sqrt((_x - light[0])**2 + (_y - light[1])**2 + h**2)
                    illum = (GtoL[1] / d**2)# * np.pi #no need, cancels with albedo / pi
                    ref[x, y, light_i] += a * (illum - ref[x, y, light_i])
        #plt.imshow(tonemap(ref), 'gray')
        #plt.show()

    path = os.path.join(os.getcwd(), 'addons', 'precomputedDL.npz')
    np.savez_compressed(path, a=lightPlacements)

@cuda.jit
def gpu_precomputedDL(refPos, lpos, out_dl, rng_states) :
    idx = cuda.grid(1)
    pos = refPos[idx]
    h = 4
    accums = 10
    x, y = pos[0], pos[1]

    for accum_i in range(1, accums + 1) :
        a = 1. / accum_i
        _x = x + xoroshiro128p_uniform_float32(rng_states, idx)
        _y = y + xoroshiro128p_uniform_float32(rng_states, idx)
        current_pos = (_x, 0, _y)
        illum = 0

        for light_i, light in enumerate(lpos) :
            GtoL = (light[0] - current_pos[0], h - current_pos[1], light[1] - current_pos[2])
            norm = math.sqrt(GtoL[0]**2 + GtoL[1]**2 + GtoL[2]**2)
            GtoL = GtoL[1] / norm
            d = math.sqrt((_x - light[0])**2 + (_y - light[1])**2 + h**2)
            illum = (GtoL / d**2)# * np.pi #no need, cancels with albedo / pi
            out_dl[int(x), int(y), light_i] += a * (illum - out_dl[x, y, light_i])

@cuda.jit
def gpu_vmask(refPos, lpos, onehot_tps, out_vmask) :
    idx = cuda.grid(1)
    samples = 128
    pos = refPos[idx]
    for light_i, light in enumerate(lpos) :
        if onehot_tps[int(pos[0]), int(pos[1]), 1] == 1 :
            out_vmask[int(pos[0]), int(pos[1]), light_i] = 0
            continue

        for sample_i in range(1, samples + 1) :
            a = 1. / sample_i
            interpX = int(light[0] + a * (pos[0] - light[0]))
            interpY = int(light[1] + a * (pos[1] - light[1]))

            isV = onehot_tps[interpX, interpY, 1] != 1

            if not isV :
                out_vmask[int(pos[0]), int(pos[1]), light_i] = 0
                break