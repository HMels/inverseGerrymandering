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
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?dl=88646
# Download the file named "CSV met statistische symbolen" for the buurten you want
# We are interested in the subjects 
#   Regiocode (gemeente)
#   Particuliere huishoudens (Aantal)
#   Opleidingsniveau/Laag/Waarde (%)
#   Opleidingsniveau/Middelbaar/Waarde (%)
#   Opleidingsniveau/Hoog/Waarde (%)
#   SES-WOA/Totaalscore/Gemiddelde score (Getal)"
inputData = InputData("Data/SES_WOA_scores_per_wijk_en_buurt_10042023_174114.csv")

# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?dl=88649
inputData.add_path("Data/SES_WOA_scores_per_wijk_en_buurt_10042023_174723.csv")

# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?dl=8864C
inputData.add_path("Data/SES_WOA_scores_per_wijk_en_buurt_10042023_175337.csv")



#%%
# Source: https://www.atlasleefomgeving.nl/kaarten
# algemene kaarten: Wijk- en buurt informatie
inputData.load_geo_data('Data/wijkenbuurten_2022_v1.GPKG')
inputData.buurt_filter(devmode=True)

## TODO for some reason a lot of data gets deleted somewhere here in multiple occasions

#%% Translate locations to a grid
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode("Amsterdam").latitude , geolocator.geocode("Amsterdam").longitude ]
inputData.map2grid(latlon0)
inputData.polygon2grid(latlon0)


#%% Save the object to a file
with open("inputData.pickle", "wb") as f:
    pickle.dump(inputData, f)