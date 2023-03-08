# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:06:36 2023

@author: Mels
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy import distance

# import modules from file
from dataset import Dataset

tf.config.run_functions_eagerly(True)  ### TODO should be False
plt.close('all')


#%% load data
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# Download the file named "CSV met statistische symbolen"
SES = pd.read_csv("Data/SES_WOA_scores_per_wijk_en_buurt_08032023_175111.csv", delimiter=';', quotechar='"', na_values='       .')
#SES = SES.fillna(0)
# Replace '       .' with NaN in columns 3 and 4 and delete these rows as they don't have SES data
SES = SES.replace('       .', float('nan'))
SES.dropna(inplace=True)
SES = SES.values

# Extract relevant columns
wijk = SES[:,1]  # Wijk (neighborhood) name
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)  # Number of private households
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)  # Socio-economic value of the region
#Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)  # Numerical measure of inequality in a region
# TODO: use Spreiding in the calculation



#%% load locations using geopandas.
import geopandas as gpd

filename = 'Data/wijkenbuurten_2022_v1.GPKG' # Source: https://www.atlasleefomgeving.nl/kaarten
gdf = gpd.read_file(filename)
gdf = gdf.to_crs('EPSG:4326') # to longlat
gdf = gdf.explode(index_parts=True)  # convert multipolygon to polygon
mycoordslist = [list(x.exterior.coords) for x in gdf.geometry] # make a list out of it


#% convert the polygons to center coordinates
center_coordinates = []
for coords in mycoordslist:
    center_coordinates.append( np.average(coords, axis=0) ) 
center_coordinates = np.array(center_coordinates)
center_coordinates[:, [0, 1]] = center_coordinates[:, [1, 0]]


#%% add indices of each buurt in order to get the coordinates
index=[]
SES_WOA_nieuw=[]
part_huishoudens_nieuw=[]
wijk_nieuw=[]
i=0
print("WARNING: GDF CAN ONLY FIND BUURTEN AND NOT WIJKEN OR CITIES. THEREFORE, A LOT OF DATA WILL BE MISSING:")
for loc in wijk:
    truefalse=False # used to check if the loc is found in the gdf file
    j = 0
    for buurt in gdf.buurtnaam:
        if buurt==loc:
            truefalse=True
            index.append(j)
            SES_WOA_nieuw.append(SES_WOA[i])
            part_huishoudens_nieuw.append(part_huishoudens[i])
            wijk_nieuw.append(wijk[i])
            break
        else:
            j+=1
            
    if not truefalse: # check if wijk was not found
        print("Warning:",loc,"has not been found in gdf data. Check if instance should have been found")
        
    i+=1
    
# save all in arrays
SES_WOA_nieuw = np.array(SES_WOA_nieuw)
part_huishoudens_nieuw = np.array(part_huishoudens_nieuw)
wijk_nieuw = np.array(wijk_nieuw)
Locs = center_coordinates[index,:]


# Translate locations to a grid
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode("Amsterdam").latitude , geolocator.geocode("Amsterdam").longitude ]
for i in range(Locs.shape[0]):
    # Store latitude and longitude in a grid
    loc = np.zeros(2,dtype=np.float32)
    loc[0] = distance.distance( latlon0 , [Locs[i,0], latlon0[1]] ).m
    loc[0] = loc[0] if (latlon0[0] < Locs[i,0]) else -loc[0]
    loc[1] = distance.distance( latlon0 , [latlon0[0], Locs[i,1]] ).m
    loc[1] = loc[1] if (latlon0[1] < Locs[i,1]) else -loc[1]
    Locs[i,:] = loc
    

#%% Define model parameters
N_communities = 5 # Number of communities
Niterations = 50 # Number of iterations for training

# Initialize model with preprocessed data
model = Dataset(SES_WOA_nieuw, part_huishoudens_nieuw, N_communities, Locs)

# Define optimization algorithm and learning rate
optimizer = tf.keras.optimizers.Adamax(learning_rate=.1)
'''
# Adagrad doesn't converge below half of the SES value
# Nadam works but has shocky convergence
# Adam results in a very uniform map
# Adamax works well ->  lr=1 fails. 
#                       lr=.1 might be too uniform
#                       lr=.01 might be better but handling of weights and reg should be better

#TODO discuss the figures I get. Compare with how good the reguralisation works and use different regs
'''

# Define variables to store costs during training
costs = np.zeros((Niterations, 5))

print("INITIAL VALUES: ")
model.print_summary()

#%% Train the model for Niterations iterations
print("OPTIMISING...")
for i in range(Niterations):
    loss_value = model.train_step(optimizer) # Take one optimization step
    if i % 10 == 0:
        # Print current loss and individual cost components
        print("Step: {}, Loss: {}".format(i, loss_value.numpy()))
        print("Partial costs:\n   SES variance = {},\n   L2 population positive = {},\n   L1 population bounds = {},\n   L1 distance = {}\n".format(
            model.SES_variance.numpy(),
            model.L2_popPositive.numpy(),
            model.L1_popBounds.numpy(),
            model.L1_distance.numpy()
        ))
    # Record cost values for plotting
    costs[i, 0] = loss_value
    costs[i, 1] = model.SES_variance.numpy()
    costs[i, 2] = model.L2_popPositive.numpy()
    costs[i, 3] = model.L1_popBounds.numpy()
    costs[i, 4] = model.L1_distance.numpy()

print("FINISHED!\n")


#%% Plot the population on a map of Amsterdam    
img = plt.imread("Data/amsterdam.PNG")
fig, ax = plt.subplots()
#ax.imshow(img, extent=[-3000, 4500, -2300, 2000])
ax.imshow(img, extent=[-3000, 4000, -2300, 3000])
ax.scatter(Locs[:, 0], Locs[:, 1], s=part_huishoudens_nieuw/100, alpha=1, label="Neighbourhoods") # Plot locations
ax.scatter(model.community_locs[:, 0], model.community_locs[:, 1], s=model.mapped_population_size/100, 
           alpha=1, label="Communities") # Plot community locations
ax.scatter(0, 0, alpha=.7) # Plot origin
ax.legend()


#%%
# Plot cost values over time
fig1, ax1 = plt.subplots()
ax1.plot(costs[:, 0]-costs[:, 2], label="Total costs", ls="-")
ax1.plot(costs[:, 1], label="SES variance", ls="--")
ax1.plot(costs[:, 3], label="L1 population bounds", ls="--")
ax1.plot(costs[:, 4], label="L1 distance", ls="--")
ax1.plot(costs[:, 2], label="L2 population positive", ls=":")
ax1.set_xlim(0, Niterations)
ax1.set_ylim(0, np.max(costs[:, 0]-costs[:, 2])*1.2)
plt.legend()


# histogram of the economic data
SES_mapped = model.mapped_socioeconomic_data.numpy()[:,0]
SES_append = np.append(SES_WOA_nieuw, SES_mapped)

fig2, ax2 = plt.subplots()
num_bins = SES_WOA.shape[0]
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)
n, bins, patches = ax2.hist(SES_WOA_nieuw, bins=bin_edges, color = 'r',edgecolor = "black",
                            alpha=0.3, label='Initial SES', density=True)
n, bins, patches = ax2.hist(SES_mapped, bins=bin_edges-bin_edges[0]/2, color = 'g',
                            alpha=0.5, label='Mapped SES', density=True)
ax2.legend(loc='upper right')
ax2.set_xlabel('SES')
ax2.set_ylabel('Frequency')
ax2.set_title('Comparison of the economic data by population')
plt.show()

print("OPTIMISED VALUES: ")
model.print_summary()