import torch
import os
import time
import lotus_postprocess
import lotus_common

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.nn.utils.parametrizations import spectral_norm
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

def sn_conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def sn_tconv2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def sn_conv3d(*args, **kwargs):
    return spectral_norm(nn.Conv3d(*args, **kwargs))

def sn_tconv3d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose3d(*args, **kwargs))

def buildKernel(kernel_size) :
    hw = 4
    k = kernel_size / 2.
    accums = 10

    light_pos = np.array([k, hw, k], dtype=np.float32)
    kernel = torch.zeros([kernel_size, kernel_size], dtype=torch.float32)

    for accum_i in range(1, accums + 1) :
        a = 1. / accum_i
        for x in range(kernel_size + 1) :
            for y in range(kernel_size + 1) :
                _x = x + np.random.rand(1)[0]
                _y = y + np.random.rand(1)[0]

                current_pos = np.array([_x, 0, _y], dtype=np.float32)
                GtoL = light_pos - current_pos
                GtoL /= np.linalg.norm(GtoL, 2)

                d = np.sqrt((_x - k)**2 + (_y - k)**2)

                if d <= k:
                    d = np.sqrt((_x - k)**2 + (_y - k)**2 + hw**2)
                    kernel[x, y] = kernel[x, y] + a * (GtoL[1] / d**2 - kernel[x, y])

    return kernel.view(1, 1, kernel_size**2)

class shuffleAug(nn.Module) :
    def __init__(self) :
        super(shuffleAug, self).__init__()

    def flipX(self, x, grid_b, grid_x, grid_y) :
        flip = torch.randint(2, (x.size(0), 1, 1), device=x.device) 
        new_grid_x = flip * (127 - grid_x) + (1 - flip) * grid_x
        return x.permute(0, 2, 3, 1).contiguous()[grid_b, new_grid_x, grid_y].permute(0, 3, 1, 2).contiguous()

    def flipY(self, x, grid_b, grid_x, grid_y) :
        flip = torch.randint(2, (x.size(0), 1, 1), device=x.device) 
        new_grid_y = flip * (127 - grid_y) + (1 - flip) * grid_y
        return x.permute(0, 2, 3, 1).contiguous()[grid_b, grid_x, new_grid_y].permute(0, 3, 1, 2).contiguous()

    def swap(self, x, grid_b, grid_x, grid_y) :
        swap = torch.randint(2, (x.size(0), 1, 1), device=x.device)     
        new_grid_x = swap * grid_y + (1 - swap) * grid_x
        new_grid_y = swap * grid_x + (1 - swap) * grid_y
        return x.permute(0, 2, 3, 1).contiguous()[grid_b, new_grid_x, new_grid_y].permute(0, 3, 1, 2).contiguous()
    
    def forward(self, *tensors) :
        grid_b, grid_x, grid_y = torch.meshgrid(
            torch.arange(tensors[0].size(0), dtype=torch.long, device=tensors[0].device),
            torch.arange(tensors[0].size(2), dtype=torch.long, device=tensors[0].device),
            torch.arange(tensors[0].size(3), dtype=torch.long, device=tensors[0].device),)
        
        x = torch.cat(tensors, dim=1)
        x = self.flipX(x, grid_b, grid_x, grid_y)
        x = self.flipY(x, grid_b, grid_x, grid_y)
        x = self.swap(x, grid_b, grid_x, grid_y)
        x = self.flipX(x, grid_b, grid_x, grid_y)
        x = self.flipY(x, grid_b, grid_x, grid_y)

        tensors_out = ()
        c_s = 0
        for tensor in tensors :
            tensors_out += (x[:, c_s:c_s + tensor.size(1), ...],)
            c_s += tensor.size(1)

        return tensors_out
    
class SPADE(nn.Module) :
    def __init__(self, in_style_features, out_features) :
        super(SPADE, self).__init__()
        self.in_style_features = in_style_features
        self.out_features = out_features
        self.hidden_dim = 64
        self.norm = nn.InstanceNorm2d(self.out_features)
        self.embedding_layer = nn.Sequential(nn.Conv2d(self.in_style_features, self.hidden_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True))
        self.netGamma = nn.Conv2d(self.hidden_dim, self.out_features, kernel_size=3, stride=1, padding=1)
        self.netBeta = nn.Conv2d(self.hidden_dim, self.out_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x, style, road_mask) :
        #style = F.interpolate(style * road_mask, size=x.size()[2:])
        style = F.interpolate(style, size=x.size()[2:])
        embedding = self.embedding_layer(style)
        gamma = self.netGamma(embedding)
        beta = self.netBeta(embedding)

        x = self.norm(x)
        out = x * (gamma + 1) + beta
        return out
    
class resSPADE(nn.Module) :
    def __init__(self, in_features, out_features, in_style_features) :
        super(resSPADE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_style_features = in_style_features

        self.spade1 = SPADE(self.in_style_features, self.in_features)
        self.spade2 = SPADE(self.in_style_features, self.out_features)
        
        self.activ = nn.LeakyReLU(0.2, True)

        self.spadeConv1 = sn_conv2d(self.in_features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.spadeConv2 = sn_conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)

        self.spadeConv3 = sn_conv2d(self.in_features, self.out_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.spade3 = SPADE(self.in_style_features, self.in_features)

    def forward(self, x, style, road_mask) :
        x = x + torch.randn_like(x)
        residual = self.spadeConv3(self.spade3(x, style, road_mask))
        x = self.activ(self.spadeConv1(self.activ(self.spade1(x, style, road_mask))))
        x = self.activ(self.spadeConv2(self.activ(self.spade2(x, style, road_mask))))
        out = x + residual
        return out
  
class spadeDecoder(nn.Module) :
    def __init__(self, in_features, init_features, in_styleFeatures, out_features) :
        super(spadeDecoder, self).__init__()
        self.in_features = in_features
        self.in_styleFeatures = in_styleFeatures
        self.out_features = out_features
        self.init_features = init_features
        nFilters = lambda lvl : self.init_features * 2**lvl

        self.linear = self.doubleConv(self.in_features, nFilters(3))
        
        self.spade8_16 = resSPADE(nFilters(3), nFilters(3), self.in_styleFeatures)
        self.spade16_32 = resSPADE(nFilters(3), nFilters(2), self.in_styleFeatures)
        self.spade32_64 = resSPADE(nFilters(2), nFilters(1), self.in_styleFeatures)
        self.spade64_128 = resSPADE(nFilters(1), nFilters(0), self.in_styleFeatures)
        self.spade128 = resSPADE(nFilters(0), nFilters(0), self.in_styleFeatures)

        self.act = nn.LeakyReLU(0.2, True)
        self.upSample = nn.Upsample(scale_factor=2)
        self.clf = sn_conv2d(nFilters(0), self.out_features, kernel_size=3, stride=1, padding=1)
    
    def doubleConv(self, in_f, out_f, mid_f = None) :
        mid_f = mid_f if mid_f != None else out_f

        return nn.Sequential(
            sn_conv2d(in_f, mid_f, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            sn_conv2d(mid_f, out_f, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),)
    
    def forward(self, x, y, road_mask) :
        latent = self.linear(x)
        x = self.upSample(self.spade8_16(latent, y, road_mask))
        x = self.upSample(self.spade16_32(x, y, road_mask))
        x = self.upSample(self.spade32_64(x, y, road_mask))
        x = self.upSample(self.spade64_128(x, y, road_mask))
        x = self.spade128(x, y, road_mask)
        return self.act(self.clf(x))

class DiscrBlock(nn.Module) :
    def __init__(self, inFeatures, outFeatures, hasSkips, scaleFactor) :
        super(DiscrBlock, self).__init__()
        self.hasSkips = hasSkips
        self.scaleFactor = scaleFactor

        if self.scaleFactor < 1 :
            self.resample = nn.AvgPool2d(2)
        else :
            self.resample = nn.Upsample(scale_factor=self.scaleFactor)

        if self.scaleFactor < 1 :
            self.mainPath = nn.Sequential(
                sn_conv2d(inFeatures, outFeatures, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                sn_conv2d(outFeatures, outFeatures, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                self.resample)
        else :
            self.mainPath = nn.Sequential(
                self.resample,
                sn_conv2d(inFeatures, outFeatures, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                sn_conv2d(outFeatures, outFeatures, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),)

        if self.hasSkips :
            if self.scaleFactor < 1 :
                self.shortcut = nn.Sequential(
                    sn_conv2d(inFeatures, outFeatures, kernel_size=1, stride=1, padding=0, bias=False),
                    self.resample)
            else :
                self.shortcut = nn.Sequential(self.resample,
                    sn_conv2d(inFeatures, outFeatures, kernel_size=1, stride=1, padding=0, bias=False))
                
    def forward(self, x) :
        if self.hasSkips :
            return self.mainPath(x) + self.shortcut(x)
        else :
            return self.mainPath(x)
      
class DiscrUnet(nn.Module) :
    def __init__(self, in_features, out_features, init_internal_f, hasSkips, outputAct=nn.Sigmoid()) :
        super(DiscrUnet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hasSkips = hasSkips
        self.init_features = init_internal_f
        numF = lambda lvl : self.init_features * 2**lvl

        self.act = nn.LeakyReLU(0.2)

        self.linearX = sn_conv2d(self.in_features, numF(0), kernel_size=3, stride=1, padding=1)
        self.linearY = sn_conv2d(self.in_features, numF(0), kernel_size=3, stride=1, padding=1)

        self.conv128_64 = DiscrBlock(numF(0), numF(1), self.hasSkips, 0.5)
        self.conv64_32 = DiscrBlock(numF(1), numF(2), self.hasSkips, 0.5) 
        self.conv32_16 = DiscrBlock(numF(2), numF(3), self.hasSkips, 0.5)
        self.conv16_8 = DiscrBlock(numF(3), numF(3), self.hasSkips, 0.5)

        self.conv8_16 = DiscrBlock(numF(3), numF(2), self.hasSkips, 2.0)
        self.conv16_32 = DiscrBlock(numF(2) + numF(3), numF(2), self.hasSkips, 2.0)
        self.conv32_64 = DiscrBlock(numF(2) + numF(2), numF(1), self.hasSkips, 2.0)
        self.conv64_128 = DiscrBlock(numF(1) + numF(1), numF(0), self.hasSkips, 2.0)
        self.last_layer = nn.Sequential(
            sn_conv2d(numF(0) + numF(0), numF(0), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.clf = sn_conv2d(numF(0), self.out_features, kernel_size=1, stride=1, padding=0)
        self.outputAct = outputAct
    
    def forward(self, x, y, road_mask, isFakeGen) :
        latentX = self.act(self.linearX(x))

        enc1 = self.conv128_64(latentX)
        enc2 = self.conv64_32(enc1)
        enc3 = self.conv32_16(enc2)
        enc4 = self.conv16_8(enc3)
        local = None

        local = torch.concat([self.conv8_16(enc4), enc3], dim=1)
        local = torch.concat([self.conv16_32(local), enc2], dim=1)
        local = torch.concat([self.conv32_64(local), enc1], dim=1)
        local = torch.concat([self.conv64_128(local), latentX], dim=1)
        local = self.last_layer(local)
        h = self.outputAct(self.clf(local))

        return h, enc4

class lightGridFilter(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, grid, mask) :
        ctx.save_for_backward(grid, mask)
        high = 400. / (2 * np.pi)
        low = 100. / (2 * np.pi)
        grid = grid * mask
        clippedGrid = grid * (grid >= low).float()
        return clippedGrid
        
    @staticmethod
    def backward(ctx, grad_out) :
        grid, mask = ctx.saved_tensors
        #assumes: 1/b * log(1 + exp(b * grid * mask))
        beta = 20.
        grad = F.sigmoid(beta * grid * mask) * mask

        return grad_out * grad, None

class Generator(nn.Module) :
    def __init__(self, config) :
        super(Generator, self).__init__()
        self.config = config
        self.noise_features = self.config['noiseDim']
        self.init_filters = self.config['generatorInitFilters']#noise_features // 2**4
        self.isBaseline = self.config['useGenBaseline']
        nFilters = lambda lvl : self.init_filters * 2**lvl
        self.kernel = None
        self.precombutedDL = None
        self.softplus = nn.Softplus()
        self.act = nn.LeakyReLU(0.2, True)
        self.gridFilter = lightGridFilter.apply
        self.baselineNoise = None

        if self.isBaseline :
            self.encoder = spadeDecoder(self.noise_features, self.init_filters, 1, nFilters(0))
        else :
            self.encoder = spadeDecoder(self.noise_features, self.init_filters, 1, nFilters(0))

        self.kw_light = 11
        self.ks_light = self.kw_light // 2
        self.pw_light = 0

        self.blockDown = nn.Sequential(
            sn_conv3d(nFilters(0), nFilters(0), kernel_size=(11, 11, 1), stride=(11, 11, 1), padding=0),
            nn.LeakyReLU(0.2, True))

        self.localHeadDown = nn.Conv2d(nFilters(0), 1, kernel_size=1, stride=1, padding=0)
        
    def getBlocks2D(self, x, x_seg) :
        roads = x_seg[:, 2:3, ::]
        x = torch.cat([x, roads], dim=1)
        b, c, h, w = x.size()

        x_chunks = F.unfold(x, kernel_size=self.kw_light, stride=self.ks_light, padding=self.pw_light, dilation=1).view(b, c, self.kw_light, self.kw_light, -1)
        goal_chunks = x_chunks[:, :-1, ::]
        road_chunks = x_chunks[:, -1:, ::]
        thresh = 1./3.
        a = road_chunks.sum((2, 3), keepdim=True) / (self.kw_light * self.kw_light)
        indicator = torch.where(a >= thresh, torch.ones_like(a), a * (1. / thresh))
        numLights = int(np.sqrt(x_chunks.size(-1)))
        #goal_chunks = goal_chunks.mean((2, 3))
        goal_chunks = self.blockDown(goal_chunks)
        goal_chunks = goal_chunks.view(b, goal_chunks.size(1), numLights, numLights)

        return goal_chunks, indicator, torch.zeros_like(road_chunks)

    def forward(self, z, onehot_tps, seg, goal, albedo, vmask) :
        if self.kernel == None :
            self.kernel = buildKernel(11)
            self.kernel = self.kernel.expand(seg.size(0), 128**2, -1)
            self.kernel = self.kernel.to(seg.device)
            self.baselineNoise = z

        if self.precombutedDL == None :
            path = os.path.join(os.getcwd(), 'addons', 'precomputedDL.npz')
            self.precombutedDL = torch.tensor(np.load(path)['a']).unsqueeze(0).permute(0, 3, 2, 1)
            self.precombutedDL = self.precombutedDL.expand(seg.size(0), -1, -1, -1)
            self.precombutedDL = self.precombutedDL.to(seg.device)

        if self.isBaseline :
            mean_goal = goal[:, 0:1, ...]
            road_mask = seg[:, 2:3, ...]
            enc = self.encoder(self.baselineNoise, mean_goal, road_mask)
        else :
            mean_goal = goal[:, 0:1, ...]
            road_mask = seg[:, 2:3, ...]
            mean_goal *= road_mask
            enc = self.encoder(z, mean_goal, road_mask)

        z = enc
        blocks, indicator, chunks = self.getBlocks2D(z, seg)
        z = self.localHeadDown(blocks).view(seg.size(0), 1, 1, 1, -1)
        z = self.gridFilter(z, indicator)
        fake_direct = torch.sum(z.squeeze(1).transpose(3, 1) * self.precombutedDL * vmask, dim=1, keepdim=True) * albedo
            
        chunks[:, :, self.kw_light//2, self.kw_light//2, :] = z[:, :, 0, 0, :]
        fake_placement = F.fold(chunks.view(goal.size(0), self.kw_light**2, -1), [128, 128], self.kw_light, 1, self.pw_light, self.ks_light)
        fake_blobs = self.kernel * fake_placement.flatten(-2, -1).transpose(1, 2)
        fake_blobs = F.fold(fake_blobs.transpose(1, 2), [128, 128], 11, dilation=1, stride=1, padding=11 // 2)
        #fake_direct /= 20.
        #fake_blobs /= 20.

        return fake_direct, fake_blobs, fake_direct, z.squeeze(), z.squeeze() * (1 - indicator)
    
    def forwardLG(self, z, seg, goal) :
        mean_goal = goal[:, 0:1, ...]
        road_mask = seg[:, 2:3, ...]
        mean_goal *= road_mask
        z = self.encoder(z, mean_goal, road_mask)
        blocks, indicator, _ = self.getBlocks2D(z, seg)
        z = self.localHeadDown(blocks).view(seg.size(0), 1, 1, 1, -1)
        z = self.gridFilter(z, indicator)
        numLights = int(np.sqrt(z.size(-1)))
        return z.view(z.size(0), 1, numLights, numLights).abs()
    
class Descirminator(nn.Module) :
    def __init__(self, in_featuresX, config) :
        super(Descirminator, self).__init__()
        self.config = config
        self.in_featuresX = in_featuresX
        self.init_filters = self.config['discriminatorInitFilters']
        self.norm_factor = self.config['discriminatorNormFactor']
        self.encoder = DiscrUnet(2, 1, self.init_filters, True)
    
    def forward(self, x, x_placement, seg, goal, onehot_tps, albedo, vmask, isFakeGen=False) :
        mean_goal = goal[:, 0:1, ...]
        x = torch.cat([x, mean_goal], dim=1)
        x /= self.norm_factor
        local_enc, _ = self.encoder(x, mean_goal, seg[:, 0:1, ...], isFakeGen)
        return local_enc

class GANLoss(nn.Module) :
    def __init__(self, loss_tp='vanilla'):
        super(GANLoss, self).__init__()
        self.loss_tp = loss_tp
        self.loss_bce = nn.BCELoss(reduction='none')

    def __call__(self) :
        pass

    def D_local_loss(self, real_output, g_output, seg, mask) :
        real_loss = self.loss_bce(real_output, mask)
        fake_loss = self.loss_bce(g_output, torch.zeros_like(g_output))
        roads = seg[:, 2:3, ...]
        real_loss = (real_loss * roads).sum(dim=(2, 3)).mean()
        fake_loss = (fake_loss * roads).sum(dim=(2, 3)).mean()
        return (real_loss + fake_loss) * 0.5, real_loss, fake_loss

    def G_local_loss(self, g_output, seg) :
        loss = self.loss_bce(g_output, torch.ones_like(g_output))
        roads = seg[:, 2:3, ...]
        loss = (loss * roads).sum(dim=(2, 3)).mean()
        return loss

class baseGAN(nn.Module) :
    def __init__(self, plotter, config) :
        super(baseGAN, self).__init__()
        self.config = config
        self.loss_tp = 'vanilla'
        self.noise_features = self.config['noiseDim']
        self.discr_training_cycle = self.config['discriminatorTrainingCycle']
        self.isBaseline = self.config['useGenBaseline']
        self.lr_G = self.config['genLearningRate']
        self.lr_D = self.config['discrLearningRate']

        self.Loss_fn = GANLoss(self.loss_tp)
        self.lambda_l1 = self.config['generatorL1Mult']
        self.plotter = plotter
        self.loss_l2 = nn.MSELoss()

    def _reset_parameters(self) : 
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def setRequiresGrads(self, model, enable) :
        for param in model.parameters():
            param.requires_grad = enable
    
    def eval_goal_err(self, x, y, goal, seg) :
        mean_goal = goal[:, 0:1, ...]

        count_per_tp = torch.sum(seg, dim=(2,3))
        denom = torch.where(count_per_tp != 0, 1. / count_per_tp, 0)
        diff = (x - mean_goal).abs()
        goal_l1_err = (diff * seg[:, 2:]).sum(dim=(2,3)) * denom[:, 2:]
        
        diff2 =  torch.where(mean_goal > 0, diff / mean_goal, torch.zeros_like(mean_goal))
        rel_l1_err = (diff2 * seg[:, 2:]).sum(dim=(2,3)) * denom[:, 2:]

        return goal_l1_err.mean().item(), rel_l1_err.mean().item()

    def eval_goal_uniformity(self, x, y, goal, seg, onehot_tps) :
        roads = seg[:, 2:, ...]
        mean_goal = goal[:, 0:1, ...]
        uniformity = torch.where(mean_goal > 0, x / mean_goal, torch.zeros_like(mean_goal))
        count_roads = torch.sum(roads, dim=(2,3))
        uniformity_count = torch.sum(uniformity >= 0.2, dim=(2, 3))
        mean_u_per = uniformity_count / count_roads

        keys, counts = torch.unique(onehot_tps, sorted=True, return_counts=True)
        mean_u = torch.zeros(1, dtype=torch.float32, device=x.device)
        mean_l = torch.zeros(1, dtype=torch.float32, device=x.device)
        mean_avg = torch.zeros(1, dtype=torch.float32, device=x.device)

        for key, value in list(zip(keys[1:], counts[1:])) :
            a = x[onehot_tps == key]
            goal_seg = mean_goal[onehot_tps == key]
            avg_illum = a.sum() / value
            min_illum = a.min()
            max_illum = a.max()
            mean_u += min_illum / avg_illum
            mean_l += min_illum / max_illum
            mean_avg += (goal_seg - a).abs().mean()
            
        mean_u /= (keys.size(0) - 1)
        mean_l /= (keys.size(0) - 1)
        mean_avg /= (keys.size(0) - 1)
        return mean_u_per.mean().item(), mean_u.mean().item(), mean_l.mean().item(), mean_avg.mean().item(), *self.eval_goal_err(x, y, goal, seg)

    def getEmptyMetricsLog(self) :
        return { 'l2_err' : 0.,
            'seg_err' : 0., 'goal_err' : 0., }
    
    def getEmptyGLossLog(self) :
        return { 'lossG_local' : 0., **self.getEmptyMetricsLog() }
    
    def getEmptyDLossLog(self) :
        return  {
            'lossD_local' : 0,
            'lossD_real_local' : 0.,
            'lossD_fake_local' : 0.,}
    
    def getEmptyLossLog(self) :
        lossD = self.getEmptyDLossLog()
        lossG = self.getEmptyGLossLog()
        return lossD, lossG
    
    def evalMetrics(self, fake_direct, real_direct, fake_y, real_y, onehot_tps, seg, x_goal) :
        roads = seg[:, 2:3, ...]
        mean_goal = x_goal[:, 0:1, ...]

        count_per_tp = torch.sum(seg, dim=(2,3))
        denom = torch.where(count_per_tp != 0, 1. / count_per_tp, 0)
        l1_err = ((fake_direct - mean_goal).abs() * roads).sum(dim=(2,3)) * denom[:, 2:]

        uniformity = torch.where(mean_goal > 0, fake_direct / mean_goal, torch.zeros_like(mean_goal))
        uniformity_err = (uniformity * roads).sum(dim=(2,3)) * denom[:, 2:]
        uniformity = torch.sum(uniformity >= 0.2, dim=(2, 3))
        mean_u = uniformity * denom[:, 2:]

        l2_err = self.loss_l2(fake_direct, real_direct)

        ret = {
            'goal_l1_err' : l1_err.mean().item(),
            'goal_uniformity' : uniformity_err.mean().item(),
            'goal_uniformity_attain' : mean_u.mean().item(),
            'l2_err' : l2_err.item(), }
        
        return ret

class streetGAN(baseGAN) :
    def __init__(self, plotter, config) :
        super(streetGAN, self).__init__(plotter, config)
        self.G_global = Generator(config)
        self.D_global = Descirminator(1, config)
        self.augModule = shuffleAug()
        self.applyAug = True
        self.useMaskedDiscr = self.config['useDiscriminatorRelError']
        self.useGeneratorLoss = self.config['useGeneratorL1Loss']
        self.discrRelThresh = self.config['discriminatorErrThresh']
        self.latentDim = 8
        self.latentSize = self.latentDim**2
        self._reset_parameters()
        self.buildOptimizers()

    def getDict(self) :
        return {
            'Gmodel_state_dict' : self.G_global.state_dict(),
            'Dmodel_state_dict' : self.D_global.state_dict(),
            'g_optimizer_state_dict' : self.optimizerG.state_dict(),
            'd_optimizer_state_dict' : self.optimizerD.state_dict(),
        }

    def loadCheckpoint(self, checkpoint, inference, config) :
        self.config = config
        self.G_global.load_state_dict(checkpoint['Gmodel_state_dict'])
        if not inference :
            self.D_global.load_state_dict(checkpoint['Dmodel_state_dict'])
            self.optimizerG.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.optimizerD.load_state_dict(checkpoint['d_optimizer_state_dict'])

    def numParams(self) :
        countFn = lambda model : sum(p.numel() for p in model.parameters() if p.requires_grad)
        Gparams = countFn(self.G_global)
        Dparams = countFn(self.D_global)

        return (Gparams, Dparams)

    def buildOptimizers(self) :
        self.optimizerG = torch.optim.Adam(self.G_global.parameters(), lr=self.lr_G, betas=(self.config['adamBeta1'], self.config['adamBeta2']), amsgrad=self.config['useAMSgrad'])
        self.optimizerD = torch.optim.Adam(self.D_global.parameters(), lr=self.lr_D, betas=(self.config['adamBeta1'], self.config['adamBeta2']), amsgrad=self.config['useAMSgrad'])
        
    def latentVecHier(self, device, batch_size, numDims, latentSize) :
        z_latents = []
        
        for level_i in range(5) :
            latentShape = latentSize * 2**level_i
            z = torch.randn(batch_size, self.noise_features * latentShape**2, dtype=torch.float32)

            if device != - 1 :
                z = z.to(device)

            z_latents += [z.view(batch_size, self.noise_features, latentShape, latentShape),]
        return z_latents
    
    def latentVec(self, device, batch_size, numDims, latentSize) :
        latentShape = latentSize * 2**0

        z = torch.randn(batch_size, numDims * latentShape**2, dtype=torch.float32)

        if device != - 1 :
            z = z.to(device)

        return z.view(batch_size, numDims, latentShape, latentShape)

    def forward(self, onehot_tps, goal, seg, albedo, vmask, z=None) :
        if z is None :
            z = self.latentVec(goal.get_device(), goal.size(0), self.noise_features, self.latentDim)
        return self.G_global(z, onehot_tps, seg, goal, albedo, vmask)

    def forwardLG(self, goal, seg, z=None) :
        if z is None :
            z = self.latentVec(goal.get_device(), goal.size(0), self.noise_features, self.latentDim)
        return self.G_global.forwardLG(z, seg, goal)
    
    def backwardD(self, real_gi, fake_gi, real_direct, fake_direct, real_blobs, fake_blobs, x_seg, x_goal, onehot_tps, albedo, vmask) :
        self.optimizerD.zero_grad()

        real_direct = torch.autograd.Variable(real_direct, requires_grad=True)
        x_goal = torch.autograd.Variable(x_goal, requires_grad=True)

        if self.applyAug :
            fake_direct, fake_blobs, real_direct, real_blobs, x_seg, x_goal, onehot_tps, albedo, vmask = self.augModule(
                fake_direct, fake_blobs, real_direct, real_blobs, x_seg, x_goal, onehot_tps, albedo, vmask)
            
        fake_local = self.D_global(fake_direct, fake_blobs, x_seg, x_goal, onehot_tps, albedo, vmask, True)
        real_local = self.D_global(real_direct, real_blobs, x_seg, x_goal, onehot_tps, albedo, vmask)

        #if self.iter_step % 100 == 0 :
            #self.plotter.plotGANBatch(recDirect, real_direct, 1, f'direct_recon_iter{self.iter_step}', f'epoch_{self.epoch_step}')

        mask = x_seg[:, 2:3, ...] #roads

        if self.useMaskedDiscr :
            mean_goal = x_goal[:, 0:1, ...]    
            dist = mean_goal - real_direct
            rel_dist = torch.where(mean_goal > 0, dist.abs() / mean_goal, torch.zeros_like(mean_goal))
            mask = mask * (rel_dist <= self.discrRelThresh).float()

        loss_discr_local, loss_discr_real_local, loss_discr_fake_local = self.Loss_fn.D_local_loss(real_local, fake_local, x_seg, mask)

        total_loss = loss_discr_local
        total_loss.backward()
        self.optimizerD.step()

        return {
            'lossD_local' : loss_discr_local.item(),
            'lossD_real_local' : loss_discr_real_local.item(),
            'lossD_fake_local' : loss_discr_fake_local.item(),}

    def backwardG(self, onehot_types, x_goal, x_seg,
        real_blobs, real_gi, real_direct,
        fake_blobs, fake_gi, fake_direct,
        fake_placement, fake_negPlacement,
        albedo, vmask) :

        if self.applyAug :
            fake_direct, fake_blobs, real_direct, real_blobs, x_seg, x_goal, onehot_types = self.augModule(
                fake_direct, fake_blobs, real_direct, real_blobs, x_seg, x_goal, onehot_types)

        total_loss = 0

        if self.isBaseline :
            roads = x_seg[:, 2:3, ...]
            total_loss = self.loss_l2_nor(fake_direct, real_direct)
            total_loss = (total_loss * roads).sum(dim=(2, 3)).mean()
            gen_local = total_loss
        else :
            fake_local = self.D_global(fake_direct, fake_blobs, x_seg, x_goal, onehot_types, albedo, vmask, True)
            gen_local = self.Loss_fn.G_local_loss(fake_local, x_seg)
            total_loss = gen_local

            if self.useGeneratorLoss:
                roads = x_seg[:, 2:3, ...]
                mean_goal = x_goal[:, 0:1, ...]
                count_per_tp = torch.sum(x_seg, dim=(2,3))
                denom = torch.where(count_per_tp != 0, 1. / count_per_tp, 0)
                fake_l1 = ((fake_direct - mean_goal).abs() * roads).sum(dim=(2,3)) * denom[:, 2:]
                total_loss += self.lambda_l1 * fake_l1.mean()

        total_loss.backward()
        self.optimizerG.step()

        with torch.no_grad() :
            val_metrics = self.evalMetrics(fake_direct, real_direct, fake_gi, real_gi, onehot_types, x_seg, x_goal)

        return {
            'lossG_local' : gen_local.item(),
            **val_metrics }

    def copy_G_params(self, model):
        flatten = deepcopy(list(p.data for p in model.parameters()))
        return flatten

    def load_params(self, model, new_param):
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)

    def optimizationStep(self, batch, shouldTrainGen, iter_step, epoch_step) :
        onehot_types, goal, albedo, seg, real_gi, real_direct, real_blobs, vmask = batch

        self.iter_step = iter_step
        self.epoch_step = epoch_step
        
        dict_lossD = self.getEmptyDLossLog()
        dict_lossG = self.getEmptyGLossLog()

        if self.isBaseline == False :
            with torch.no_grad() :
                fake_gi, fake_blobs, fake_direct, _, _ = self.forward(onehot_types, goal, seg, albedo, vmask)

            self.setRequiresGrads(self.D_global, True)

            dict_lossD = self.backwardD(real_gi, fake_gi, real_direct, fake_direct, real_blobs, fake_blobs, seg, goal, onehot_types, albedo, vmask)

        if shouldTrainGen :
            self.setRequiresGrads(self.D_global, False)
            self.optimizerG.zero_grad()
            fake_gi, fake_blobs, fake_direct, fake_placement, fake_negPlacement = self.forward(onehot_types, goal, seg, albedo, vmask)

            dict_lossG = self.backwardG(onehot_types, goal, seg,
                real_blobs, real_gi, real_direct,
                fake_blobs, fake_gi, fake_direct,
                fake_placement, fake_negPlacement, albedo, vmask)

        return dict_lossD, dict_lossG
    
    def validationStep(self, batch, iter_step, epoch_step, numSamples) :
        onehot_types, goal, albedo, seg, real_gi, real_direct, real_blobs, vmask = batch

        self.iter_step = iter_step
        self.epoch_step = epoch_step
        total_val_metrics = None

        with torch.no_grad() :
            for sample_i in range(numSamples) :
                fake_gi, fake_blobs, fake_direct, _, _ = self.forward(onehot_types, goal, seg, albedo, vmask)
                val_metrics = self.evalMetrics(fake_direct, real_direct, fake_gi, real_gi, onehot_types, seg, goal)
                if total_val_metrics == None :
                    total_val_metrics = val_metrics
                else :
                    for key in total_val_metrics.keys() :
                        total_val_metrics[key] += val_metrics[key]

        return { k : v / numSamples for k, v in total_val_metrics.items() }
    
    def postprocessLightGrid(self, gen_lightGrid, seg, placementKernel=None) :
        assert gen_lightGrid.size() == (1, 1, 24, 24)
        assert seg.size() == (1, 3, 128, 128)

        dev = gen_lightGrid.device

        clusteredGrid = gen_lightGrid[0, 0].cpu().numpy()
        clusteredGrid = torch.tensor(lotus_postprocess.clusterLights(clusteredGrid)).to(dev)
        clusteredGrid = clusteredGrid.view_as(gen_lightGrid)

        clustered_placement, _ = lotus_postprocess.gridToImage(
            clusteredGrid.view(1, 1, 1, 1, -1).to(dev),
            placementKernel)

        snapped_placement = lotus_postprocess.snapLights(
            clustered_placement[0, 0].cpu().numpy(),
            seg[0, 2:3, ...].cpu().numpy())

        snapped_placement = torch.tensor(snapped_placement).to(dev).view_as(clustered_placement)

        return snapped_placement

    def generateSnappedLights(self, onehot_types, goal, seg, albedo, vmask, precombutedDL, placementKernel) :
        with torch.no_grad() :
            gen_lightGrid = self.forwardLG(goal, seg)

        #assert lightGrid.size() == (1, 1, 24, 24)
        clusteredGrid = None
        clustered_placement = None
        snapped_placement = None

        clusteredGrid = gen_lightGrid[0, 0].cpu().numpy()
        clusteredGrid = torch.tensor(lotus_postprocess.clusterLights(clusteredGrid)).to(goal.device)
        clusteredGrid = clusteredGrid.view_as(gen_lightGrid)

        clustered_placement, clustered_blobs = lotus_postprocess.gridToImage(
            clusteredGrid.view(1, 1, 1, 1, -1).to(goal.device),
            placementKernel)

        snapped_placement = lotus_postprocess.snapLights(
            clustered_placement[0, 0].cpu().numpy(),
            seg[0, 2:3, ...].cpu().numpy())

        snapped_placement = torch.tensor(snapped_placement).to(goal.device).view_as(clustered_placement)

        gen_placement, gen_blobs = lotus_postprocess.gridToImage(
            gen_lightGrid.view(1, 1, 1, 1, -1),
            placementKernel)

        return gen_lightGrid, gen_blobs, clusteredGrid, clustered_blobs, snapped_placement

    def computeDLfromPlacement(self, lightsMap, albedo, seg, goal, onehot_tps, kernel=None) :
        dev = lightsMap.device
        blobs = None

        lightsMap_cpu = lightsMap[0].cpu().permute(2, 1, 0).numpy()
        seg = seg[0].cpu().permute(2, 1, 0).numpy()
        albedo = albedo[0].cpu().permute(2, 1, 0)

        numLights = (lightsMap_cpu > 0).sum().item()
        geomTerm = np.zeros(shape=(128, 128, numLights), dtype=np.float32)
        vmap = np.ones(shape=(128, 128, numLights), dtype=np.float32)

        x = np.arange(start=0, stop=128, step=1)
        y = np.arange(start=0, stop=128, step=1)
        ref_pos = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        light_indices = np.where(lightsMap_cpu > 0)
        light_pos = np.array(list(zip(light_indices[0].ravel(), light_indices[1].ravel())), dtype=np.int32)
        lightArr = torch.tensor(lightsMap_cpu[light_indices])

        d_vmap = cuda.to_device(vmap)
        d_onehot = cuda.to_device(seg)
        d_light_pos = cuda.to_device(light_pos)
        d_ref_pos = cuda.to_device(ref_pos)
        lotus_common.gpu_vmask[128, 128](d_ref_pos, d_light_pos, d_onehot, d_vmap)
        cuda.synchronize()
        vmap = d_vmap.copy_to_host()

        d_geomterm = cuda.to_device(geomTerm)
        rng_states = create_xoroshiro128p_states(128 * 128, seed=1337)
        lotus_common.gpu_precomputedDL[128, 128](d_ref_pos, d_light_pos, d_geomterm, rng_states)
        cuda.synchronize()
        geomTerm = d_geomterm.copy_to_host()


        minVal = (100. / (2 * np.pi))
        img_size = goal.size(-1)
        lightIntensities = lightArr.view(1, -1, 1, 1).to(dev)
        geomTerm = torch.tensor(geomTerm).permute(2, 1, 0).reshape(1, -1, img_size, img_size).to(dev)
        vmap = torch.tensor(vmap).permute(2, 1, 0).reshape(1, -1, img_size, img_size).to(dev)
        albedo = albedo.permute(2, 1, 0).reshape(1, -1, img_size, img_size).to(dev)
        seg =  torch.tensor(seg).permute(2, 1, 0).reshape(1, -1, img_size, img_size).to(dev)

        if True :
            lightIntensities = lotus_postprocess.optimizeLights( \
                lightIntensities, goal[:, 0:1, ...], \
                minVal, geomTerm, vmap, albedo, seg[:, 2:3, ...], onehot_tps)
        
        directLighting = torch.sum(lightIntensities * geomTerm * vmap, dim=1, keepdim=True) * albedo
        lightIntensities = lightIntensities.view(-1)
        lightsMap_cpu[light_indices] = lightIntensities.cpu().numpy()
        lightMap = torch.tensor(lightsMap_cpu).permute(2, 1, 0).reshape(1, 1, img_size, img_size).to(dev)

        if kernel != None :
            blobs = kernel * lightMap.flatten(-2, -1).transpose(1, 2)
            blobs = F.fold(blobs.transpose(1, 2), [128, 128], 11, dilation=1, stride=1, padding=11 // 2)

        return directLighting, blobs, lightIntensities, lightMap

    def generate(self, onehot_types, goal, seg, albedo, vmask, z=None) :
        with torch.no_grad():
            z = z if z != None else self.latentVec(goal.get_device(), goal.size(0), self.noise_features, self.latentDim)
            _, fake_blobs, fake_direct, _, _ = self.forward(onehot_types, goal, seg, albedo, vmask, z)

        return fake_blobs, fake_direct
    
    def generateN(self, onehot_types, goal, seg, albedo, vmask, numSamples, optimized=False) :
        fake_direct_list = []
        fake_blobs_list = []

        for sample_i in range(numSamples) :
            if optimized :
                fake_direct = self.generateOptDL(goal, seg, albedo, onehot_types)
                fake_direct_list += [fake_direct,]
            else :
                fake_blobs, fake_direct = self.generate(onehot_types, goal, seg, albedo, vmask)
                fake_direct_list += [fake_direct,]
                fake_blobs_list += [fake_blobs,]

        return fake_direct_list, fake_blobs_list

    def generateOptDL(self, goal, seg, albedo, onehot_tps) :
        with torch.no_grad() :
            gen_lightGrid = self.forwardLG(goal, seg)

        snapped_lights = self.postprocessLightGrid(gen_lightGrid, seg)
        snapped_direct, _, _, _ = self.computeDLfromPlacement(snapped_lights, albedo, seg, goal, onehot_tps)

        return snapped_direct

    def generateLG(self, goal, seg) :
        with torch.no_grad() :
            return self.forwardLG(goal, seg)

    def generateOptLights(self, goal, seg, albedo, onehot_tps) :
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad() :
            #cuda_start.record()
            gen_lightGrid = self.forwardLG(goal, seg)
            #cuda_end.record()
        #torch.cuda.synchronize()
        #print(f'{cuda_start.elapsed_time(cuda_end)}')

        s = time.time()
        cuda_start.record()
        snapped_lights = self.postprocessLightGrid(gen_lightGrid, seg)
        _, _, _, optPlacement = self.computeDLfromPlacement(snapped_lights, albedo, seg, goal, onehot_tps)
        e = time.time()
        cuda_end.record()
        torch.cuda.synchronize()
        print(f'{(e - s) * 1000}, {cuda_start.elapsed_time(cuda_end)}')
        return optPlacement