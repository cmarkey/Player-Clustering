import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("f_clustering_metrics.csv")

# normalizing variables
scaler = StandardScaler()
df['Shot Index'] = scaler.fit_transform(df[["Shot Index"]])
df['PSA Index'] = scaler.fit_transform(df[["PSA Index"]])
df['Passing Index'] = scaler.fit_transform(df[["Passing Index"]])
df['Entry Index'] = scaler.fit_transform(df[["Entry Index"]])
df['Danger Pass Index'] = scaler.fit_transform(df[["Danger Pass Index"]])
df['Danger Shot Index'] = scaler.fit_transform(df[["Danger Shot Index"]])
df['Takeaways Index'] = scaler.fit_transform(df[["Takeaways Index"]])
df['Puck Recovery Index'] = scaler.fit_transform(df[["Puck Recovery Index"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=6, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df[['Shot Index', 'PSA Index', 'Passing Index', 'Entry Index', 'Danger Pass Index',
                       'Danger Shot Index', 'Takeaways Index', 'Puck Recovery Index']])
df.loc[:, 'labels'] = kmeans.labels_
df.to_csv('kmeans_results_f.csv', index=False)

index_table_f = df.groupby('labels').agg(['mean'])
index_table_f.columns = index_table_f.columns.droplevel(1)
index_table_f = index_table_f.round(decimals=3)
index_table_f['Count'] = [16, 56, 9, 76, 1, 3]

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=index_table_f.values, colLabels=index_table_f.columns, rowLabels=index_table_f.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.show()


df2 = pd.read_csv("d_clustering_metrics.csv")

# normalizing variables
df2['Shot Index'] = scaler.fit_transform(df2[["Shot Index"]])
df2['PSA Index'] = scaler.fit_transform(df2[["PSA Index"]])
df2['Passing Index'] = scaler.fit_transform(df2[["Passing Index"]])
df2['Entry Index'] = scaler.fit_transform(df2[["Entry Index"]])
df2['Danger Pass Index'] = scaler.fit_transform(df2[["Danger Pass Index"]])
df2['Danger Shot Index'] = scaler.fit_transform(df2[["Danger Shot Index"]])
df2['Takeaways Index'] = scaler.fit_transform(df2[["Takeaways Index"]])
df2['Puck Recovery Index'] = scaler.fit_transform(df2[["Puck Recovery Index"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df2[['Shot Index', 'PSA Index', 'Passing Index', 'Entry Index', 'Danger Pass Index',
                       'Danger Shot Index', 'Takeaways Index', 'Puck Recovery Index']])
df2.loc[:, 'labels'] = kmeans.labels_
df2.to_csv('kmeans_results_d.csv', index=False)

index_table_d = df2.groupby('labels').agg(['mean'])
index_table_d.columns = index_table_d.columns.droplevel(1)
index_table_d = index_table_d.round(decimals=3)
index_table_d['Count'] = [60, 27, 1]

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=index_table_d.values, colLabels=index_table_d.columns, rowLabels=index_table_d.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.show()

# kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 26, }
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
#     kmeans.fit(df[['Shot Index', 'PSA Index', 'Passing Index', 'Entry Index', 'Danger Pass Index', 'Danger Shot Index',
#                    'Takeaways Index', 'Puck Recovery Index']])
#     sse.append(kmeans.inertia_)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
#
# kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
# print(kl.elbow)
#
# silhouette_coefficients = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(df[['Shot Index', 'PSA Index', 'Passing Index', 'Entry Index', 'Danger Pass Index', 'Danger Shot Index',
#                    'Takeaways Index', 'Puck Recovery Index']])
#     score = silhouette_score(df[['Shot Index', 'PSA Index', 'Passing Index', 'Entry Index', 'Danger Pass Index',
#                                  'Danger Shot Index', 'Takeaways Index', 'Puck Recovery Index']], kmeans.labels_)
#     silhouette_coefficients.append(score)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()
