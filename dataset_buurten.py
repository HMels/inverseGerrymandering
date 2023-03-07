# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:33:28 2023

@author: Mels
"""
import geopandas as gpd
import numpy as np
import pandas as pd

filename = 'Data/wijkenbuurten_2022_v1.GPKG'
gdf = gpd.read_file(filename)
gdf.to_crs      # data in coordinates
gdf = gdf.explode(index_parts=True)  # convert multipolygon to polygon
mycoordslist = [list(x.exterior.coords) for x in gdf.geometry] # make a list out of it

# average the coordinates to get the centerpoint 
avg_coords = []
for coords in mycoordslist:
    avg_coords.append( np.average(coords, axis=0) ) 
avg_coords = np.array(avg_coords)

#%%
# Load SES data
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# Download the file named "CSV met statistische symbolen"
SES = pd.read_csv("Data/SES_WOA_scores_per_wijk_en_buurt_06032023_210307.csv", delimiter=';',quotechar='"').values

# Extract relevant columns
wijk = SES[:,1]  # Wijk (neighborhood) name
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)  # Number of private households
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)  # Socio-economic value of the region
Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)  # Numerical measure of inequality in a region

#%% add indices of each buurt in order to get the coordinates
index=[]
SES_WOA_nieuw=[]
wijk_nieuw=[]
i=0
for loc in wijk:
    truefalse=False
    j = 0
    #loc = loc.replace("buurt","")
    #loc = loc.replace("West","")
    #loc = loc.replace("Zuid","")
    #loc = loc.replace("Noord","")
    #loc = loc.replace("Oost","")
    for buurt in gdf.buurtnaam:
        if buurt==loc:
            #print(loc,", ", buurt)
            truefalse=True
            index.append(j)
            SES_WOA_nieuw.append(SES_WOA[i])
            wijk_nieuw.append(wijk[i])
            break
        else:
            j+=1
            
    if False: # check if wijk was not found
        print(loc)
        
    i+=1
    
SES_WOA_nieuw = np.array(SES_WOA_nieuw)
wijk_nieuw = np.array(wijk_nieuw)
            
#%% 
for buurt in gdf.buurtnaam:
    if buurt.__contains__('wijk'):
        print(buurt)