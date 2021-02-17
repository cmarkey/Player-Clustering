import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#importing and formating data for PCA
full_indexes = pd.read_csv(r'clustering_metrics.csv')[['Player', 'Position', 'Team', 'Shot Index',
            'PSA Index', 'Passing Index', 'Entry Index', 'Danger Pass Index', 'Danger Shot Index',
            'Takeaways Index', 'Puck Recovery Index']]
prepped_indexes = full_indexes[['Shot Index', 'PSA Index', 'Passing Index', 'Entry Index',
            'Danger Pass Index', 'Danger Shot Index','Takeaways Index',
            'Puck Recovery Index']].to_numpy()
category_names = ['Shot Index','PSA Index', 'Passing Index', 'Entry Index',
            'Danger Pass Index', 'Danger Shot Index','Takeaways Index',
            'Puck Recovery Index']

#normalizing data for PCA
prepped_indexes= (prepped_indexes-prepped_indexes.mean())/prepped_indexes.std()


#checking histograms
for i in range(len(prepped_indexes.T)):
    plt.hist(prepped_indexes.T[i])
    plt.title(category_names[i]+" Distribution")
    plt.show()


# plotting some slices of data and their principle axes
component_pca = PCA().fit(prepped_indexes)
print(component_pca.components_)
print(component_pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

for i in range(len(prepped_indexes.T)-1):
    plt.scatter(prepped_indexes[:, i], prepped_indexes[:, i+1], alpha=0.2)
    for length, vector in zip(component_pca.explained_variance_[i:i+2], component_pca.components_[i:i+2]):
        v = vector * 3 * np.sqrt(length)
        draw_vector(component_pca.mean_[i:i+2], component_pca.mean_[i:i+2] + v[i:i+2])
    plt.axis('equal');
    plt.show()


#choosing n_components
plt.plot(np.cumsum(component_pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
#choosing 3 components to retain at 95% of explained variance. Chosen at the 'knee'


#actual principal component analysis
pca = PCA(n_components=3).fit(prepped_indexes)
indexes_reduced = pca.transform(prepped_indexes)
index_df = pd.DataFrame(indexes_reduced, columns=['Var1','Var2', 'Var3'])
index_df['Player'] = full_indexes[['Player']].copy(deep=True)
index_df['Position'] = full_indexes[['Position']].copy(deep=True)
index_df['Team'] = full_indexes[['Team']].copy(deep=True)
index_df.to_csv('transformed_metrics.csv')
#print(index_df)
#print(full_indexes)
print("original shape:   ", prepped_indexes.shape)
print("transformed shape:", indexes_reduced.shape)

#histogram of results for each variable
for i in range(len(indexes_reduced.T)):
    plt.hist(indexes_reduced.T[i])
    plt.show()

#cross sectional slices of results
plt.scatter(indexes_reduced[:,0], indexes_reduced[:,1],alpha=.2)
plt.title("Variable 1 vs Variable 2")
plt.show()

plt.scatter(indexes_reduced[:,1], indexes_reduced[:,2],alpha=.2)
plt.title("Variable 2 vs Variable 3")

plt.scatter(indexes_reduced[:,0], indexes_reduced[:,2],alpha=.2)
plt.title("Variable 1 vs Variable 3")
