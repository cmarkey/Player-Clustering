import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Forwards
df = pd.read_csv("f_transformed_metrics.csv")
df_normal = pd.read_csv("f_clustering_metrics.csv")

# normalizing variables
scaler = StandardScaler()
df['Var1'] = scaler.fit_transform(df[["Var1"]])
df['Var2'] = scaler.fit_transform(df[["Var2"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=5, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df[['Var1', 'Var2']])
df.loc[:, 'labels'] = kmeans.labels_

df.loc[df['Player'] == "Valeria Pavlova", 'labels'] = 1

plt.style.use("fivethirtyeight")
plt.scatter("Var1", "Var2", data=df, c="labels",cmap="tab10")
plt.xlabel("Var1")
plt.ylabel("Var2")
plt.title("Forward")
plt.show()
df.to_csv('kmeanspca_results_f.csv', index=False)

df_normal = df_normal.merge(df, on='Player', how='left')
index_table_f = df_normal.groupby('labels').agg(['mean'])
index_table_f.columns = index_table_f.columns.droplevel(1)
index_table_f = index_table_f.round(decimals=3)
index_table_f['Count'] = [82, 26, 44, 9]

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=index_table_f.values, colLabels=index_table_f.columns, rowLabels=index_table_f.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.show()


# Defenseman

df2 = pd.read_csv("d_transformed_metrics.csv")
df_normal2 = pd.read_csv("d_clustering_metrics.csv")

# normalizing variables
df2['Var1'] = scaler.fit_transform(df2[["Var1"]])
df2['Var2'] = scaler.fit_transform(df2[["Var2"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df2[['Var1', 'Var2']])
df2.loc[:, 'labels'] = kmeans.labels_

plt.style.use("fivethirtyeight")
plt.scatter("Var1", "Var2", data=df2, c="labels",cmap="tab10")
plt.xlabel("Var1")
plt.ylabel("Var2")
plt.title("Defense")
plt.show()
df2.to_csv('kmeanspca_results_d.csv', index=False)

df_normal2 = df_normal2.merge(df2, on='Player', how='left')
index_table_d = df_normal2.groupby('labels').agg(['mean'])
index_table_d.columns = index_table_d.columns.droplevel(1)
index_table_d = index_table_d.round(decimals=3)
index_table_d['Count'] = [28, 17, 43]

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
#     kmeans.fit(df[['Var1', 'Var2']])
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
#     kmeans.fit(df[['Var1', 'Var2']])
#     score = silhouette_score(df[['Var1', 'Var2']], kmeans.labels_)
#     silhouette_coefficients.append(score)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()