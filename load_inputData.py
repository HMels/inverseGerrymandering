# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:34:58 2023

Load and save the inputData

@author: Mels
"""
import pickle
from geopy.geocoders import Nominatim

from inputData import InputData

#%% Load data
if True: # loading the geoData takes too long so this way I only have to do it once
    # Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
    # Download the file named "CSV met statistische symbolen"
    inputData = InputData("Data/SES_WOA_scores_per_wijk_en_buurt_08032023_175111.csv")
    
    
    # Source: https://www.atlasleefomgeving.nl/kaarten
    inputData.load_geo_data('Data/wijkenbuurten_2022_v1.GPKG')
    inputData.buurt_filter(loadGeometry=True)
    
else: 
    inputData.reload_path("Data/SES_WOA_scores_per_wijk_en_buurt_08032023_175111.csv")
    inputData.buurt_filter(loadGeometry=True)


#%% Translate locations to a grid
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode("Amsterdam").latitude , geolocator.geocode("Amsterdam").longitude ]
inputData.map2grid(latlon0)
inputData.polygon2grid(latlon0)


#%% Save the object to a file
with open("inputData.pickle", "wb") as f:
    pickle.dump(inputData, f)