# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:41:17 2023

@author: Mels
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from matplotlib.patches import Polygon as PolygonPatch

# import modules from file
from inputData import InputData
from modelGeo import ModelGeo

def create_color_dict(N):
    """
    Creates a dictionary of colors with RGB values that are evenly spaced.
    
    Parameters:
    - N (int): The number of colors to generate.
    
    Returns:
    - colors_dict (dict): A dictionary of colors with keys ranging from 0 to N-1 and values in the format
                          recognized by matplotlib.pyplot.
    """
    cmap = plt.cm.get_cmap('gist_rainbow', N)  # Get a colormap with N colors
    
    colors_dict = {}
    for i in range(N):
        rgb = cmap(i)[:3]  # Extract the RGB values from the colormap
        color = np.array(rgb) * 255  # Convert the RGB values from [0, 1] to [0, 255]
        colors_dict[i] = '#{:02x}{:02x}{:02x}'.format(*color.astype(int))  # Convert the RGB values to hex format
    
    return colors_dict



#%% Load data
if False: # loading the geoData takes too long so this way I only have to do it once
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


#%% Define model parameters
N_communities = 7 # Number of communities
N_iterations = 50 # Number of iterations for training

# Define optimization algorithm and learning rate
optimizer = tf.keras.optimizers.Adamax(learning_rate=.25)
model = ModelGeo(inputData, N_communities, N_iterations, optimizer)
'''
# Adagrad doesn't converge below half of the SES value
# Nadam works but has shocky convergence
# Adam results in a very uniform map
# Adamax works well ->  lr=1 fails. 
#                       lr=.1 might be too uniform
#                       lr=.01 might be better but handling of weights and reg should be better

#TODO discuss the figures I get. Compare with how good the reguralisation works and use different regs
'''
SES_initial = model.mapped_Socioeconomic_data.numpy()

print("INITIAL VALUES: ")
model.print_summary()



#%% plot initial state
#cdict = {0: 'c', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5:}
cdict = create_color_dict(N_communities)
colour = []
for label in model.labels.numpy():
    colour.append(cdict[label])

fig, ax = plt.subplots()
extent=[-2000, 2000, -1200, 2800]   
for i, polygon in enumerate(model.InputData.GeometryGrid):
    patch = PolygonPatch(np.array(polygon.exterior.xy).T, facecolor=colour[i], alpha=0.5)
    ax.add_patch(patch)
 
colors = [cdict[i] for i in range(model.Communities.N)]
ax.scatter(model.Communities.Locations[:,0], model.Communities.Locations[:,1],
           s=model.Communities.Population/100, c=colors, alpha=.8, ec='black')    

ax.set_xlim(extent[0],extent[1])
ax.set_ylim(extent[2],extent[3])
ax.set_title('Communities Before Optimization')
    

# histogram of the economic data 
#TODO, delete this initial state
fig2, ax2 = plt.subplots()
num_bins = model.InputData.Socioeconomic_data.shape[0]
SES_append = np.append(model.InputData.Socioeconomic_data, model.Communities.Socioeconomic_data)
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)

n, bins, patches = ax2.hist(model.InputData.Socioeconomic_data.numpy(), bins=bin_edges, color = 'r',edgecolor = "black",
                            alpha=0.3, label='Initial neighbourhoods', density=True)
n, bins, patches = ax2.hist(SES_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.4, label='Initial communities', density=True)

ax2.legend(loc='upper right')
ax2.set_xlabel('SES')
ax2.set_ylabel('Frequency')
ax2.set_title('Comparison of the economic data by population')
plt.show()


#%% Train the model for Niterations iterations
print("OPTIMISING...")
model.refine(N_iterations, temperature=0)
model.applyMapCommunities()
print("FINISHED!\n")


#%% Polygon Plot    
# reload colours
colour = []
for label in model.labels.numpy():
    colour.append(cdict[label])
    
fig, ax = plt.subplots()   
for i, polygon in enumerate(model.InputData.GeometryGrid):
    patch = PolygonPatch(np.array(polygon.exterior.xy).T, facecolor=colour[i], alpha=0.5)
    ax.add_patch(patch)
    
colors = [cdict[i] for i in range(model.Communities.N)]
ax.scatter(model.Communities.Locations[:,0], model.Communities.Locations[:,1],
           s=model.Communities.Population/100, c=colors, alpha=.8, ec='black')

ax.set_xlim(extent[0],extent[1])
ax.set_ylim(extent[2],extent[3])
ax.set_title('Communities After Optimization')


#%% Plotting
fig1, ax1 = model.OptimizationData.plotCosts()

# histogram of the economic data 
fig2, ax2 = plt.subplots()
num_bins = model.InputData.Socioeconomic_data.shape[0]
SES_append = np.append(model.InputData.Socioeconomic_data, model.Communities.Socioeconomic_data)
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)

n, bins, patches = ax2.hist(model.InputData.Socioeconomic_data.numpy(), bins=bin_edges, color = 'r',edgecolor = "black",
                            alpha=0.3, label='Initial neighbourhoods', density=True)
n, bins, patches = ax2.hist(SES_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.4, label='Initial communities', density=True)
n, bins, patches = ax2.hist(model.Communities.Socioeconomic_data.numpy(), bins=bin_edges-bin_edges[0]/2, color = 'g',
                            alpha=0.5, label='Mapped', density=True)

ax2.legend(loc='upper right')
ax2.set_xlabel('SES')
ax2.set_ylabel('Frequency')
ax2.set_title('Comparison of the economic data by population')
plt.show()

print("OPTIMISED VALUES: ")
model.print_summary()
