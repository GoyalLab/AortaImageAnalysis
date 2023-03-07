import numpy as np

def centeroidPoints(arr):
    length = arr.shape[0]
    sum_y = np.sum(arr[:, 0])
    sum_x = np.sum(arr[:, 1])
    return sum_y/length, sum_x/length

def clusterCoord(cluster):
    coord = []
    clusterSize =[]
    for key in cluster:
        points = np.asarray([df[df['label'] == cluster[key][i]][['centroid-0', 'centroid-1']].to_numpy()[0] for i in range(len(cluster[key]))])
        centroid = centeroidPoints(points)
        coord.append(centroid)
        clusterSize.append(len(cluster[key]))
    return np.asarray(coord), np.asarray(clusterSize)