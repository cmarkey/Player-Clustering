import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

#no significant results :(

def hdbscan_clustering(data_file,non_transformed_file,min_points):
    #reading in and prepping data
    raw_data = pd.read_csv(non_transformed_file)
    raw_cols = list(raw_data.columns)
    raw_cols.append('Archetype')
    full_data = pd.read_csv(data_file)
    stripped_data = full_data.drop(columns=['Player','Position','Team'])
    metric_names = list(stripped_data.columns)
    metric_names.append('Archetype')
    stripped_data = stripped_data.to_numpy()

    #clustering
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_points, gen_min_span_tree=True)
    archetype_groups = clustering.fit_predict(stripped_data)
    if len(np.unique(archetype_groups)) == 1:
        return 'No clustering found'
    #clustering visualization in first 2 dimensions
    plt.scatter(stripped_data[:,0], stripped_data[:,1],alpha=.2,c=archetype_groups)
    plt.title(metric_names[0]+" vs "+metric_names[1])
    plt.show()

    #cluster analysis
    merge_1 = pd.DataFrame(np.concatenate((stripped_data, np.array([archetype_groups]).T),
                                axis=1), columns=metric_names)
    #print(merge_1)
    merge_2 = pd.merge(full_data, merge_1, on=metric_names[0:-1], how='left').filter(items=['Player','Position','Team','Archetype'])
    #print(merge_2)
    full_clustered_data = pd.merge(raw_data, merge_2, on=['Player','Position','Team'], how='left')
    #print(full_clustered_data)
    stat_summaries = pd.DataFrame({'Avg Shot Index' : [], 'Avg PSA Index' : [],
                                    'Avg Passing Index' : [], 'Avg Entry Index' : [],
                                    'Avg Danger Pass Index' : [], 'Avg Danger Shot Index' : [],
                                    'Avg Takeaways Index' : [], 'Avg Puck Recovery Index' : []})
    column_names = ['Avg Shot Index', 'Avg PSA Index', 'Avg Passing Index', 'Avg Entry Index',
                                    'Avg Danger Pass Index', 'Avg Danger Shot Index',
                                    'Avg Takeaways Index', 'Avg Puck Recovery Index']
    #the actual summary statistic code. Very messy, use at your own risk
    for i in range(len(np.unique(archetype_groups))):
        archetype_group = full_clustered_data.drop(full_clustered_data[full_clustered_data['Archetype'] == np.unique(archetype_groups)[i]].index).reset_index(drop=True)
        #print(archetype_group.mean())
        archetype_stats = pd.DataFrame(np.array([archetype_group.mean()])[:,:-1], columns=column_names)
        stat_summaries = stat_summaries.append(archetype_stats, ignore_index=True)
        #print(stat_summaries)
    return stat_summaries

#forwards
#hdbscan with non-pca data
forward_plain = hdbscan_clustering('f_clustering_metrics.csv','f_clustering_metrics.csv',5) #5 is max for finding any clusters
print(forward_plain)
#hdbscan with pca data
forward_pca = hdbscan_clustering('f_transformed_metrics.csv','f_clustering_metrics.csv',15) #15 is max for finding any clusters
print(forward_pca)

#defensemen
#hdbscan with non-pca data
defensemen_plain = hdbscan_clustering('d_clustering_metrics.csv','d_clustering_metrics.csv',8) #8 is max for finding any clusters
print(defensemen_plain)
#hdbscan with pca data
defensemen_pca = hdbscan_clustering('d_transformed_metrics.csv','d_clustering_metrics.csv',9) #9 is max for finding any clusters
print(defensemen_pca)
