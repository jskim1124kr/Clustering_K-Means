import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def second_smallest(numbers):
    a1, a2 = float('inf'), float('inf')

    for x in numbers:
        if x <= a1:
            a1, a2 = x, a1
        elif x < a2:
            a2 = x
    return a2

# Randomly initialization
def initialize_centroids(X, k,iter):
    rand_postion = []
    for i in range(K):
        rand_postion.append(random.choice(X))
    init_centroids = np.array(rand_postion)
    print(str(iter+1) + " step's random initialization")
    print(init_centroids)
    return init_centroids

def assigned_centroid(X, centroids):
    K = centroids.shape[0]

    m = X.shape[0]
    mat_dist = np.zeros((m, K))

    for i in range(K):
        v = centroids[i, :]
        diff = X - v
        dist = np.sum(diff*diff, axis=1)
        mat_dist[:, i] = dist

    idx = np.argmin(mat_dist, 1)
    return idx

def update_centroid(X, idx, K):
    M, N = X.shape
    centroids = np.zeros((K, N))

    for i in range(K):
        sel = (idx == i)
        centroids[i, :] = np.mean(X[sel, :], axis=0)

    return centroids

def distortion_function(X, centroids, idx, K):
    cost = 0

    for i in range(K):
        sel = (idx == i)
        cluster_points = X[sel, :]

        for j in range(len(cluster_points)):
            cost += np.linalg.norm(cluster_points[j] - centroids[i])

    return cost

def K_means(X, init_centroids, iteration):
    K = init_centroids.shape[0]
    centroids = init_centroids

    idx = assigned_centroid(X, centroids)

    centroids = update_centroid(X, idx, K)
    cost = distortion_function(X, centroids, idx, K)
    print("Cost : ",cost)

    return centroids, cost

dataFrame = pd.read_csv('Iris.csv')
data = dataFrame.values

X = data[:, 1:-1].astype(float)

K = 3

costs = []
iters = []
best_cost = 9999

best_cost = list()
best_centroid = list()

for n in range(1, 5):
    cost_list = list()
    centroid_list = list()

    for i in range(100):
        init_centroids = initialize_centroids(X, n, i)

        centroids,cost = K_means(X, init_centroids, iteration)
        cost = (1.0 / len(X)) * cost

        cost_list.append(cost)
        centroid_list.append(centroids)

    min_cost = np.argmin(cost_list)

    if (np.isnan(cost_list[min_cost]) == True):
        small = second_smallest(cost_list)
        min_cost = cost_list.index(small)
        best_cost.append(cost_list[min_cost])
        best_centroid.append(centroid_list[min_cost])
    else:
        best_cost.append(cost_list[min_cost])
        best_centroid.append(centroid_list[min_cost])

x_axis = list()
for i in range(1, 5):
    x_axis.append(i)


plt.plot(x_axis, best_cost, 'b')
plt.grid()
plt.legend()
plt.xlabel('Number of Clusters')
plt.ylabel('Cost Rate')
plt.show()
