# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:33:28 2023

@author: Mels
"""

import geopandas as gpd
filename = 'Data/wijkenbuurten_2022_v1.GPKG'
gdf = gpd.read_file(filename)

#%%
import numpy as np
import pandas as pd

# Load SES data
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# Download the file named "CSV met statistische symbolen"
SES = pd.read_csv("Data/SES_WOA_scores_per_wijk_en_buurt_10022023_163026.csv", delimiter=';',quotechar='"').values

# Extract relevant columns
wijk = SES[:,1]  # Wijk (neighborhood) name
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)  # Number of private households
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)  # Socio-economic value of the region
Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)  # Numerical measure of inequality in a region

#%%
for buurt in gdf.buurtnaam:
    if buurt.__contains__('Kazernebuurt'):
        print(buurt)