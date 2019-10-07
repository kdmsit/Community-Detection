# region Packages
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# endregion

if __name__ == '__main__':
    # Read The file
    inputfilePath = "../data/11_twoCirclesData.csv"
    twocircledataframe=pd.read_csv(inputfilePath).values
    x = twocircledataframe[:, 0]
    y = twocircledataframe[:, 1]
    #Generate Adjacency Matrix
    adj_matrix=np.zeros((len(twocircledataframe),len(twocircledataframe)))
    sigma=0.04472135955
    for i in range(len(twocircledataframe)):
        for j in range(len(twocircledataframe)):
            distance=np.square((np.linalg.norm(twocircledataframe[i]-twocircledataframe[j])))
            adj_matrix[i][j]=np.exp(-distance/np.square(sigma))
    #Generate Graph From Adj Matrix
    G = nx.from_numpy_matrix(adj_matrix)
    node_list = list(G.node)
    # Compute Normalised Laplacian
    laplacian_Matrix = nx.normalized_laplacian_matrix(G).todense()
    # Fincout Eigen Values and Eigen Vectors of Normalised Laplacian
    eigenValues, eigenVectors = np.linalg.eigh(laplacian_Matrix)
    EigV = eigenVectors.T
    # Sort Eigen Values
    sortedEigenValueIndex = np.argsort(eigenValues)
    #Take Bottom two eigen Vectors and Stack them vertically.
    eigenVector1 = EigV[sortedEigenValueIndex[1]]
    eigenVector2 = EigV[sortedEigenValueIndex[2]]
    emb =np.vstack((eigenVector1, eigenVector2))
    # Apply K-Means on Embedding Matrix
    kmeans = KMeans(n_clusters=2).fit(emb.T)
    labels = kmeans.labels_
    # Plot Network Clusters
    nodeColorMap = []
    count0=0
    count1 = 0
    for i in range(len(node_list)):
        if (labels[i] == 0):
            nodeColorMap.append('blue')
            count0 = count0 + 1
        elif (labels[i] == 1):
            nodeColorMap.append('red')
            count1 = count1 + 1
    plt.scatter(x, y,c=nodeColorMap, alpha=0.5)
    print("Spectral Clustering Done.....")
    print("Inner Circle Size : " + str(min(count0,count1)) + ", Outer Circle Size :" + str(max(count0,count1)))
    plt.title('Scatter plot of two circle problem after spectral clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('spectralClustering_twocircle.png');
    print('spectralClustering_twocircle.png figure saved in home directory')
    plt.show()
    plt.close();