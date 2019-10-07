# region Packages
import numpy as np
import sys
import os
import math
import networkx as nx
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
from numpy import linalg as LA
# endregion

if __name__ == '__main__':
    #Read The gml file
    inputfilePath = "../data/dolphins/dolphins.gml"
    G = nx.Graph()
    #Compute the graph
    G=nx.read_gml(inputfilePath)
    #Compute Adjacency Matrix
    Adjacency_lists=G.adj
    node_list = list(G.node)
    #Compute Normalised Laplacian
    A = nx.normalized_laplacian_matrix(G)
    laplacian_Matrix=A.todense()
    #Fincout Eigen Values and Eigen Vectors of Normalised Laplacian
    eigenValues, eigenVectors = np.linalg.eigh(laplacian_Matrix)
    EigV=eigenVectors.T
    #Sort Eigen Values
    sortedEigenValueIndex = np.argsort(eigenValues)
    #Findout Fiedler vector
    secondSmallestEigenValue = eigenValues[sortedEigenValueIndex[1]]
    secondSmallestEigenVector = EigV[sortedEigenValueIndex[1]]
    print("The Fiedler vector is as follows: ")
    print(secondSmallestEigenVector)
    #Apply K-Means on Fiedler vector
    kmeans=KMeans(n_clusters=2).fit(np.asanyarray(secondSmallestEigenVector).reshape(-1, 1))
    labels=kmeans.labels_
    #Plot Network Clusters
    nodeColorMap = []
    count=0
    for i in range(len(node_list)):
        if (labels[i] == 0):
            nodeColorMap.append('blue')
            count=count+1
        elif(labels[i] == 1):
            nodeColorMap.append('red')
    print("Spectral Clustering Done.....")
    print("One Community Size : "+str(count)+", Other Community size:"+str(62-count))
    nx.draw_networkx(G, node_color=nodeColorMap, with_labels=True)
    plt.title('Dolphin Network after spectral clustering')
    plt.savefig('spectralClustering_dolphin.png');
    print('spectralClustering_dolphin.png figure saved in home directory')
    plt.show()
    plt.close();
