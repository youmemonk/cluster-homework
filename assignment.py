import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from keras.datasets import mnist
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()


def initData(xtrain, ytrain):
    x = np.empty((0, 28, 28), dtype=np.uint8)
    y = np.empty((0), dtype=np.uint8)
    for digit in range(0, 10):
        index = np.where(ytrain == digit)[0]
        ranIdx = np.random.choice(index, 100, replace=False)
        x = np.concatenate((x, xtrain[ranIdx]), axis=0)
        y = np.concatenate((y, ytrain[ranIdx]), axis=0)

    shuffle = np.random.permutation(len(x))
    xsort = x[shuffle]
    ysort = y[shuffle]
    return (xsort, ysort)


def preprocess(x):
    m = x.shape[0]
    x = np.reshape(x, (m, 784))
    x = x.astype(float)/255
    return x


np.random.seed(0)
Xtrain, Ytrain = initData(xtrain, ytrain)
Xtrain = preprocess(Xtrain)


def optimisation(X, cluster, centroid):
    m = len(X)
    J = 0
    for i in range(m):
        dist = np.linalg.norm(X[i]-centroid[cluster[i]])
        J += (dist**2)
    J = (J/m)
    return J


def kCluster(Xtrain, K, random_init=True):
    m = Xtrain.shape[0]
    itr = 0
    if random_init:
        centroid = np.random.rand(K, 784)
    else:
        idx = np.random.randint(m, size=K)
        centroid = Xtrain[idx, :]
    cluster = [-1 for i in range(m)]
    Jprev = 0
    while True:
        cluster_group = {i: [] for i in range(K)}
        for j in range(m):
            min_dist = np.inf
            for k in range(K):
                dist = np.linalg.norm(centroid[k]-Xtrain[j])
                if(dist < min_dist):
                    min_dist = dist
                    cluster[j] = k
            cluster_group[cluster[j]].append(Xtrain[j].tolist())

        for k in range(K):
            if (len(cluster_group[k]) != 0):
                centroid[k] = np.array(cluster_group[k]).mean(axis=0)
            else:
                id = np.random.randint(m)
                centroid[k] = Xtrain[id]
        J = optimisation(Xtrain, cluster, centroid)
        itr += 1
        if (abs(J-Jprev) < 0.8):
            break
        Jprev = J

    return cluster, centroid, itr


cluster, centroid, itr = kCluster(Xtrain, 20, False)
J = optimisation(Xtrain, cluster, centroid)
print(itr, J)

centroidmg = np.reshape(centroid, (20, 28, 28))
fig = plt.figure(figsize=(15, 10))
columns = 4
rows = 5
for k in range(1, 21):
    fig.add_subplot(rows, columns, k)
    plt.imshow(centroidmg[k-1], cmap='gray')
plt.show()


def labelCluster(cluster, ytrain, K):
    data = {i: [] for i in range(K)}
    for i, c in enumerate(cluster):
        data[c].append(ytrain[i])

    label = []
    for k in range(K):
        d = Counter(data[k])
        label.append(d.most_common(1)[0][0])
    return label


def assignCluster(x_test, centroid):
    m = len(x_test)
    K = len(centroid)
    cluster = [-1 for i in range(m)]
    for j in range(m):
        min_dist = np.inf
        for k in range(K):
            dist = np.linalg.norm(centroid[k]-x_test[j])
            if (dist < min_dist):
                min_dist = dist
                cluster[j] = k
    return cluster


def accuracy(predicted_cluster, cluster_label, actual_label):
    m = len(actual_label)
    correct = 0
    for i in range(m):
        if(cluster_label[predicted_cluster[i]] == actual_label[i]):
            correct += 1
    print("Accuracy: {}%".format(correct*100/m))


idx = np.random.choice(len(xtest), 50, replace=False)
X_test, Y_test = xtest[idx], ytest[idx]
X_test = preprocess(X_test)

test_cluster = assignCluster(X_test, centroid)
cluster_label = labelCluster(cluster, Ytrain, 20)
accuracy(test_cluster, cluster_label, Y_test)

print("K    Jclust")
for K in range(5, 21):
    cluster, centroid, itr = kCluster(Xtrain, K, False)
    J = optimisation(Xtrain, cluster, centroid)
    print("{}   {}".format(K, J))
