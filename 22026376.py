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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

countryList = ['Brazil', 'Russian Federation', 'India', 'China', 'South Africa']


def transformDf(df, removeColumns, indexColumns):
    
    df=df.drop(removeColumns, axis=1)
    df.set_index(indexColumns, inplace=True)
    df.sort_index(inplace=True)
    return df

def transformDf2(df):
    
    df = df.fillna(0)
    df=df.T
    return df

def fillterCountries(df):
    
    d={}
    for x in countryList:
        d[x] = df.loc[:, x]
    return d



df = pd.read_csv("API_19_DS2_en_csv_v2_5361599.csv", skiprows=4)
#Discovering data using .describe()
df.describe()
#Redundant Columns to be removed
removeColumns=['Country Code','Indicator Code']
indexColumns=["Country Name", "Indicator Name"]


new_df = transformDf(df, removeColumns, indexColumns)
new_df.describe()


#transpose the data frame
new_df1=transformDf2(new_df)
new_df1.describe()

#saving the filtered data
x = fillterCountries(new_df1)



#country by its variable
Brazil = x['Brazil']
Russian_Federation = x['Russian Federation']
India = x['India']
China = x['China']
South_Africa = x['South Africa']


#Brazil
B_df = pd.DataFrame(Brazil)
B_df = df.skew()
B_df = B_df.index, B_df.loc[[('1990', '1995', '2000', '2005', '2010', '2015'),('Population, total', 
          'Electric power consumption (kWh per capita)', 
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)', 
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from coal sources (% of total)',
          'Access to electricity (% of population)')]]
B_df = pd.DataFrame(B_df)
B_df = B_df.rename(columns = {0:"Brazil"})
B_df = B_df.T

#Russian_Federation
R_df = pd.DataFrame(Russian_Federation)
R_df = df.skew()
R_df = R_df.index, R_df.loc[[('1990', '1995', '2000', '2005', '2010', '2015'),('Population, total', 
          'Electric power consumption (kWh per capita)',         
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)', 
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from coal sources (% of total)',
          'Access to electricity (% of population)')]]
R_df = pd.DataFrame(R_df)
R_df = R_df.rename(columns = {0:"Russian_Federation"})
R_df = R_df.T

#India
I_df = pd.DataFrame(India)
I_df = df.skew()
I_df = I_df.index, I_df.loc[[('1990', '1995', '2000', '2005', '2010', '2015'),('Population, total', 
          'Electric power consumption (kWh per capita)',
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)', 
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from coal sources (% of total)',
          'Access to electricity (% of population)')]]
I_df = pd.DataFrame(I_df)
I_df = I_df.rename(columns = {0:"India"})
I_df = I_df.T

#china
C_df = pd.DataFrame(China)
C_df = df.skew()
C_df = C_df.index, C_df.loc[[('1990', '1995', '2000', '2005', '2010', '2015'),('Population, total', 
          'Electric power consumption (kWh per capita)',
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)', 
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from coal sources (% of total)',
          'Access to electricity (% of population)')]]
C_df = pd.DataFrame(C_df)
C_df = C_df.rename(columns = {0:"China"})
C_df = C_df.T

#South africa
S_df = pd.DataFrame(South_Africa)
S_df = df.skew()
S_df = S_df.index, S_df.loc[[('1990', '1995', '2000', '2005', '2010', '2015'),('Population, total', 
          'Electric power consumption (kWh per capita)',          
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)', 
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from coal sources (% of total)',
          'Access to electricity (% of population)')]]
S_df = pd.DataFrame(S_df)
S_df = S_df.rename(columns = {0:"South_Africa"})
S_df = S_df.T

all_data = pd.concat([B_df, R_df, I_df, C_df, S_df])
#Renaming column names
all_data = all_data.rename(columns = {'Population, total': "population", 
          'Electric power consumption (kWh per capita)': "total power consumption",
          'Renewable electricity output (% of total electricity output)':"renewable source",
          'Electricity production from oil sources (% of total)':"oil sources", 
          'Electricity production from nuclear sources (% of total)':"nuclear sources",
          'Electricity production from coal sources (% of total)':"coal sources",
          'Access to electricity (% of population)':"Access to electricity"
                                      })
all_data=all_data.T
print(all_data)

# HEATMAP
Brazil.describe()
#Filtering and saving France's different years and different parameter.
heat_map_parameters = ['Population, total', 
          'Electric power consumption (kWh per capita)',          
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)', 
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from coal sources (% of total)',
          'Access to electricity (% of population)']

Br_df = Brazil.loc[:, Brazil.columns.isin(heat_map_parameters)]
#Fr_df.fillna(0, inplace=True)

#calculating correlation for France
corr_brics_heat_map = Br_df.corr()

print('xxx')
print(corr_brics_heat_map)
print('xxx')
print(Br_df)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_brics_heat_map, cmap='coolwarm', annot=True, linewidths=0.5, annot_kws={'size': 10})
plt.title('Brics')
ct.map_corr(Br_df)
plt.savefig('heatmap.png',
             dpi=400,
             bbox_inches ="tight",
             pad_inches = 1,
             transparent = False,
             orientation ='landscape')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()


