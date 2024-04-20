import torch
import numpy as np
from scipy import spatial, ndimage
from torch.nn import functional as F

def roadEdges(roads, sobel=False):
    if sobel:
        sobel_h = ndimage.sobel(roads, 0)  # horizontal gradient
        sobel_v = ndimage.sobel(roads, 1)  # vertical gradient
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        #magnitude *= 255.0 / np.max(magnitude)  # normalization
        return magnitude
    else:
        roads_grad = np.gradient(roads)
        roads_h = np.absolute(roads_grad[0])
        roads_v = np.absolute(roads_grad[1])
        magnitude = roads_h + roads_v
        return magnitude
    
def snapLights(lights, roads):
    light_coords = np.where(lights > 0)
    light_coords = list(zip(light_coords[0].ravel(), light_coords[1].ravel()))

    road_edges_grad  = roadEdges(roads.squeeze(), sobel=False)
    road_edges = road_edges_grad

    road_edge_coords = np.where(road_edges > 0)
    road_edge_coords = list(zip(road_edge_coords[0].ravel(), road_edge_coords[1].ravel()))
    road_edge_tree = spatial.KDTree(road_edge_coords)

    query_points = np.array(light_coords)
    dists, idx = road_edge_tree.query(query_points, k = 1)

    lights_snapped = np.zeros(lights.shape, dtype=np.float32).squeeze()
    for index, value in enumerate(road_edge_tree.data[idx].astype(int)):
        row, col = value
        lights_snapped[row, col] = lights[light_coords[index]]
    
    return lights_snapped

def clusterLights(lights, kappa=5, distance_upper_bound=1.5) :
    lights_clustered = lights

    while True :
        lights_clustered_cur = np.zeros(lights.shape, dtype=np.float32)
        light_coords = np.where(lights_clustered > 0)
        light_coords = list(zip(light_coords[0].ravel(), light_coords[1].ravel()))
        light_tree = spatial.KDTree(light_coords)

        query_points = np.array(light_coords)
        dists, idx = light_tree.query(query_points, k=kappa, distance_upper_bound=distance_upper_bound)

        idx2 = np.arange(query_points.shape[0])[:,np.newaxis]
        idx2 = np.repeat(idx2, kappa, axis=1)
        idx[dists == np.inf] = idx2[dists == np.inf] 

        results = light_tree.data[idx]
        result = results[0].astype(int)
        foundUpdate = False
        numRounds = 0
        for result in results:
            center_coords = (result[0][0].astype(int), result[0][1].astype(int))
            center = lights_clustered[center_coords]
            max_lum = center
            max_coords = center_coords
            for row, col in result[1:].astype(int):
                lum = lights_clustered[row, col]
                if lum > max_lum:
                    max_lum = lum
                    max_coords = (row, col)
                    foundUpdate = True
            lights_clustered_cur[max_coords] = max_lum

        numRounds += 1
        lights_clustered = np.copy(lights_clustered_cur)
        if not foundUpdate :
            print(f'Num. clustering rounds {numRounds}')
            break

    return lights_clustered
      
def optimizeLights(lightGrid, mean_goal, minValue, precombutedDL, vmask, albedo, roads, onehot_tps) :
    assert lightGrid.dim() == mean_goal.dim() == precombutedDL.dim() == vmask.dim() == albedo.dim() == roads.dim()
    assert lightGrid.size(1) == precombutedDL.size(1) == vmask.size(1)
    assert precombutedDL.size()[2:] == vmask.size()[2:] == albedo.size()[2:] == roads.size()[2:]
    assert albedo.size(1) == roads.size(1) == 1
    assert mean_goal.size(1) == 1
    #assert torch.any(torch.logical_and(lightGrid > 0, lightGrid < minValue))

    lightGrid.requires_grad = True
    lightGrid.retain_grad()
    optimizer = torch.optim.Adam([lightGrid], lr=10.0, amsgrad=True)
    stop_crit = 1.e-3
    num_steps = 100
    prev_loss = None

    count_per_tp = torch.sum(onehot_tps, dim=(2,3))
    denom = torch.where(count_per_tp != 0, 1. / count_per_tp, 0)

    for step_i in range(num_steps) :
        optimizer.zero_grad()
        lightSelectionI = (lightGrid >= minValue).float()
        x = lightGrid * lightSelectionI + (1 - lightSelectionI) * minValue
        fake_direct = torch.sum(x * precombutedDL * vmask, dim=1, keepdim=True) * albedo

        loss = (fake_direct - mean_goal)**2
        loss = ((loss * roads).sum(dim=(2, 3))).squeeze()
        loss.backward(retain_graph=True)
        #print(f'Step: {step_i} - loss: {loss.item():.10f}')

        optimizer.step()        

        if step_i == 0 :
            prev_loss = loss.item()
        else :
            if (loss - prev_loss).abs() <= stop_crit :
                break

            prev_loss = loss.item()

    print(f'Number of opt. steps {step_i}')
    #assert torch.any(torch.logical_and(lightGrid > 0, lightGrid < minValue))
    lightSelectionI = (lightGrid >= minValue).float()
    lightGrid = lightGrid * lightSelectionI + (1 - lightSelectionI) * minValue
    return lightGrid.detach()

def gridToImage(lightGrid, placementKernel=None):
    lightsMap  = torch.zeros_like(torch.empty(1, 1, 128, 128)).to(lightGrid.device)
    chunks = F.unfold(lightsMap, kernel_size=11, stride=5, padding=0, dilation=1).view(1, 1, 11, 11, -1)
    kw_light = 11
    chunks[:, :, kw_light//2, kw_light//2, :] = lightGrid[:, :, 0, 0, :]
    lightsMap = F.fold(chunks.view(1, kw_light**2, -1), [128, 128], kw_light, 1, 1, 5)

    blobs = None
    
    if placementKernel != None :
        blobs = placementKernel * lightsMap.flatten(-2, -1).transpose(1, 2)
        blobs = F.fold(blobs.transpose(1, 2), [128, 128], 11, dilation=1, stride=1, padding=11 // 2)
    
    return lightsMap, blobs
