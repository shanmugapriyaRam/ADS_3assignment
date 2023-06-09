# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:14:55 2023

@author: shanm
"""
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import numpy as np


#Data cleaning functions
def transformDf(df):
    
    rem_cols = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df.drop(rem_cols, axis=1, inplace=True)
    df = df.fillna(0)
    return df


def transformDf2(df, indicators, Country):
    
    df = df[df['Indicator Name'].isin(indicators)]
    df = df[df['Country Name'] == Country]
    return df


def transformDf3(df):
    del_col = 'Country Name'
    df.drop(del_col, axis=1, inplace=True)
    df = df.T
    df.columns = df.iloc[0]
    df.drop("Indicator Name", inplace = True)
    return df


def line(x, m, c):
    y = m*x+c
    return y 

#Initial program
df = pd.read_csv("API_19_DS2_en_csv_v2_5361599.csv", skiprows=4)

df1 = transformDf(df)

indicators = ['Population, total',
              'Electricity production from oil sources (% of total)',
              'Electricity production from nuclear sources (% of total)',
              'Electricity production from coal sources (% of total)',
              'Electric power consumption (kWh per capita)',
              'CO2 emissions from solid fuel consumption (kt)',
              'CO2 emissions from liquid fuel consumption (kt)',
              'CO2 emissions from gaseous fuel consumption (kt)',
              'Energy use (kg of oil equivalent per capita)',              
              'Cereal yield (kg per hectare)',
              'Agricultural land (sq. km)']

Country = 'Japan'

sr_df = transformDf2(df1, indicators, Country)

Japan_df = transformDf3(sr_df)
print(Japan_df.describe())
df_fit = Japan_df[["Population, total",
                   "Energy use (kg of oil equivalent per capita)"]].copy()

df_fit, df_min, df_max = ct.scaler(df_fit)
print(df_fit.describe())
print()

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))
    
nc = 3 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_fit)     

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(df_fit["Population, total"], 
            df_fit["Energy use (kg of oil equivalent per capita)"], 
            c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xc = cen[:,0]
yc = cen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=90)
# c = colour, s = size

plt.xlabel("Total Population")
plt.ylabel("Energy use (kg of oil equivalent per capita)")
plt.title("Cluster representation of Energy usage by total population")

plt.show()

#Line Fit for total population


x = df_fit["Population, total"]
y= df_fit["Energy use (kg of oil equivalent per capita)"]
popt, pcorr = opt.curve_fit(line, x, y)
print("Fit parameter", popt)
# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))


z = line(x, *popt)


plt.figure()
plt.title("Population vs Energy use")
plt.scatter(x, y, label="Population")
plt.plot(x, z, label="Energy use")
plt.xlabel("Population")
plt.ylabel("Energy use (kg of oil equivalent per capita)")
# plot error ranges with transparency
#plt.fill_between(years, lower, upper, alpha=0.5)

plt.legend(loc="upper left")
plt.show()


##electricity production 
df_fit = Japan_df[["Electricity production from oil sources (% of total)",
                   "Electricity production from nuclear sources (% of total)"]].copy()

df_fit, df_min, df_max = ct.scaler(df_fit)
print(df_fit.describe())
print()

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))
    
nc = 5 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_fit)     

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(df_fit["Electricity production from oil sources (% of total)"], 
            df_fit["Electricity production from nuclear sources (% of total)"], 
            c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xc = cen[:,0]
yc = cen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=90)
# c = colour, s = size

plt.xlabel("Electricity production from oil sources ")
plt.ylabel("Electricity production from nuclear sources ")
plt.title("cluster for electricity production")
plt.show()








