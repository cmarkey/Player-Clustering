import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Forwards
df = pd.read_csv("f_transformed_metrics.csv")

# normalizing variables
scaler = StandardScaler()
df['Var1'] = scaler.fit_transform(df[["Var1"]])
df['Var2'] = scaler.fit_transform(df[["Var2"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=4, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df[['Var1', 'Var2']])
df.loc[:, 'labels'] = kmeans.labels_

plt.style.use("fivethirtyeight")
plt.scatter("Var1", "Var2", data=df, c="labels",cmap="tab10")
plt.xlabel("Var1")
plt.ylabel("Var2")
plt.title("Forward")
plt.show()
df.to_csv('kmeanspca_results_f.csv', index=False)

#Defenseman

df2 = pd.read_csv("d_transformed_metrics.csv")

# normalizing variables
df2['Var1'] = scaler.fit_transform(df2[["Var1"]])
df2['Var2'] = scaler.fit_transform(df2[["Var2"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=4, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df2[['Var1', 'Var2']])
df2.loc[:, 'labels'] = kmeans.labels_

plt.style.use("fivethirtyeight")
plt.scatter("Var1", "Var2", data=df2, c="labels",cmap="tab10")
plt.xlabel("Var1")
plt.ylabel("Var2")
plt.title("Defense")
plt.show()
df.to_csv('kmeanspca_results_d.csv', index=False)


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