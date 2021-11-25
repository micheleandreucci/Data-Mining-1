#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("HR_Employee_MissingValuesFilled.csv", skipinitialspace=True, sep=',')


cat_df = df.select_dtypes(include='object')
for col in df.columns:
    if col in cat_df.columns.tolist():
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(le.classes_)
    else:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])

df.describe()

print(df.head())

print(df.info())

to_discard = ['Attrition', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'EnviromentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'TrainingTimesLastYear', 'Over18', 'StandardHours', 'YearsWithCurrManager', 'Age']

to_df = [col for col in df.columns if col not in to_discard]

df = df[to_df]

print(df.info())


plt.figure(figsize = (15,4))
sb.boxplot(data = df, orient = "h")
plt.show()

plt.figure(figsize = (15,6))
sb.heatmap( df.corr(), annot=True)


scatter_matrix = scatter_matrix(df, figsize=(15, 15))
for ax in scatter_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 10, rotation = 10)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 10, rotation = 0)
plt.show()

sse_list = list()
silouette_scores = {}
max_k = 50
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
    kmeans.fit(df)

    sse = kmeans.inertia_
    sse_list.append(sse)

    labels_k = kmeans.labels_
    silouette = metrics.silhouette_score(df, labels_k)
    silouette_scores[k] = silouette

plt.plot(range(2, len(sse_list) + 2), sse_list)
plt.ylabel('SSE', fontsize=22)
plt.xlabel('K', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

plt.figure(figsize = (16,5))
plt.plot(silouette_scores.values())
plt.xticks(range(2, len(sse_list) + 2), silouette_scores.keys())
plt.title("Silhouette Metric")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.show()

kl = KneeLocator(
    range(2, len(sse_list) + 2), sse_list, curve="convex", direction="decreasing"
)
print("Number of cluster: ", kl.elbow)

kmeans = KMeans(n_clusters=5, n_init=10, max_iter=100)
kmeans.fit(df)

final_sse = kmeans.inertia_
print("Final SSE: ", final_sse)

labels_k = kmeans.labels_
final_silouette = metrics.silhouette_score(df, labels_k)
print("Final silouette: ", final_silouette)
print("First 5 labels: ", kmeans.labels_[:5])
print("Dimensions of clusters: ", np.unique(kmeans.labels_, return_counts=True))

hist, bins = np.histogram(kmeans.labels_,
                          bins=range(0, len(set(kmeans.labels_)) + 1))
print(dict(zip(bins, hist)))

print(kmeans.cluster_centers_)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 4))
for i in range(0, len(centers)):
    plt.plot(centers[i], marker='o', label='Cluster %s' % i)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.xticks(range(0, len(df.columns)), df.columns, fontsize=10, rotation=20)
plt.legend(fontsize=8)
plt.show()

fig,ax = plt.subplots(4,3, figsize=(9,9))
sb.distplot(df['DailyRate'], ax = ax[0,0])
sb.distplot(df['DistanceFromHome'], ax = ax[0,1])
sb.distplot(df['HourlyRate'], ax = ax[0,2])
sb.distplot(df['MonthlyIncome'], ax = ax[1,0])
sb.distplot(df['MonthlyRate'], ax = ax[1,1])
sb.distplot(df['NumCompaniesWorked'], ax = ax[1,2])
sb.distplot(df['PercentSalaryHike'], ax = ax[2,0])
sb.distplot(df['TotalWorkingYears'], ax = ax[2,1])
sb.distplot(df['YearsAtCompany'], ax = ax[2,2])
sb.distplot(df['YearsInCurrentRole'], ax = ax[3,0])
sb.distplot(df['YearsSinceLastPromotion'], ax = ax[3,1])
plt.tight_layout()
plt.show()

df["cluster"] = kmeans.labels_
g=sb.pairplot(data = df, hue = "cluster", palette = "Accent_r")
for ax in g.axes.flatten():
    # rotate x axis labels
    ax.set_xlabel(ax.get_xlabel(), fontsize= 10 , rotation = 10)
    # rotate y axis labels
    ax.set_ylabel(ax.get_ylabel(), fontsize= 10, rotation = 0)
    # set y labels alignment
    ax.yaxis.get_label().set_horizontalalignment('right')
plt.show()

