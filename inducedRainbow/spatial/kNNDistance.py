import numpy as np
from random import seed
from random import sample
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Delaunay

def randomkNNDistances(nucIDs,len, df):
    selectedRandom = sample(nucIDs, len)
    dfRandom = df[df['label'].isin(selectedRandom)]
    nucCoordRandom = dfRandom[['centroid-0', 'centroid-1']].to_numpy()
    nbrsRandom = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(nucCoordRandom)
    distancesRand, indicesRandom = nbrsRandom.kneighbors(nucCoordRandom)
    return distancesRand

def kNN(array, neighbor):
    nbrs = NearestNeighbors(n_neighbors=neighbor, algorithm='ball_tree').fit(array)
    distances, indices = nbrs.kneighbors(array)
    return nbrs, distances, indices