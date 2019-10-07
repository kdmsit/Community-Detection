# region Packages
import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt
# endregion

def calculate_modularity(small_cluster,cluster,G):
    '''
    In this procedure we merge small_cluster with the cluster and find Modularity of the graph
    '''
    for k in range(len(small_cluster)):
        item = small_cluster[k]
        key = list(item.keys())
        item[key[0]] = cluster
        small_cluster[k] = item
    merged_partition = small_cluster + partition_dict[cluster]
    new_partition_list = []
    for j in range(len(partition_dict)):
        if j != i:
            if j == cluster:
                for item in merged_partition:
                    new_partition_list.append(item)
            else:
                for item in partition_dict[j]:
                    new_partition_list.append(item)
    new_partition_dict = {}
    for item in new_partition_list:
        new_partition_dict.update(item)
    modularity_cluster = community.modularity(new_partition_dict, G)
    return modularity_cluster

if __name__ == '__main__':
    # Read The gml file
    inputfilePath = "../data/dolphins/dolphins.gml"
    G = nx.Graph()
    # Compute the graph
    G=nx.read_gml(inputfilePath)
    nodelist = list(G.node)
    #Find best partition
    partition = community.best_partition(G)

    #Generate Dictionary based on partition label
    noofpartitions = int(len(set(partition.values())))
    partition_dict={}
    for node,label in partition.items():
        if label not in partition_dict.keys():
            partition_dict[label]=[{node:label}]
        else:
            partition_dict[label].append({node:label})

    #Compute Partition sizes
    partition_sizes = []
    for i in range(len(partition_dict)):
        partition_sizes.append(len(partition_dict[i]))
    #Find Top two partitions
    partition_indices=np.argsort(partition_sizes)
    top_two_cluster_indices=partition_indices[-2:]

    #Greddy algorithm to merge partition
    for i in range(len(partition_dict)):
        if i not in top_two_cluster_indices:
            cluster1=top_two_cluster_indices[0]
            cluster2 = top_two_cluster_indices[1]
            small_cluster1 = partition_dict[i]
            modularity_cluster1=calculate_modularity(small_cluster1,cluster1,G)
            small_cluster2 = partition_dict[i]
            modularity_cluster2 = calculate_modularity(small_cluster2, cluster2,G)
            if modularity_cluster1>modularity_cluster2:
                partition_dict[i]=small_cluster1
            else:
                partition_dict[i] = small_cluster2

    #Organise nodelist community as per final partition list.
    node_community_list={}
    for i in range(len(partition_dict)):
        for item in partition_dict[i]:
            node_community_list.update(item)
    nodelist_comm={}
    for node in nodelist:
        for keynode,community in node_community_list.items():
            if(node==keynode):
                nodelist_comm[node]=community
    community_list=[]
    for node,community in nodelist_comm.items():
        community_list.append(community)

    #Plot the network.
    set_community=set(community_list)
    node_color_list=[]
    count=0
    for comm in community_list:
        if(comm == list(set_community)[0]):
            count=count+1
            node_color_list.append('blue')
        if (comm == list(set_community)[1]):
            node_color_list.append('red')
    print("Louvain Method Clustering is Done..")
    print("One Community Size : " + str(count) + ", Other Community size:" + str(62 - count))
    plt.title('Dolphin Network after Louvain Method Clustering')
    nx.draw_networkx(G, node_color=node_color_list, with_labels=True)
    plt.savefig('Louvain Method Clustering_dolphin.png');
    print('Louvain Method Clustering_dolphin.png figure saved in home directory')
    plt.show()
    plt.close();