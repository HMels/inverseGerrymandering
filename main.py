# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:41:17 2023

@author: Mels
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# import modules from file
from model import Model
from inputData import InputData



#%% Load data
if True: # loading the geoData takes too long so this way I only have to do it once
    # Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
    # Download the file named "CSV met statistische symbolen"
    inputData = InputData("Data/SES_WOA_scores_per_wijk_en_buurt_08032023_175111.csv")
    
    
    # Source: https://www.atlasleefomgeving.nl/kaarten
    inputData.load_geo_data('Data/wijkenbuurten_2022_v1.GPKG')
    inputData.buurt_filter()
    
else: 
    inputData.reload_path("Data/SES_WOA_scores_per_wijk_en_buurt_08032023_175111.csv")
    inputData.buurt_filter()


#%% Translate locations to a grid
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode("Amsterdam").latitude , geolocator.geocode("Amsterdam").longitude ]
inputData.map2grid(latlon0)


#%% Define model parameters
N_communities = 5 # Number of communities
N_iterations = 50 # Number of iterations for training

# Define optimization algorithm and learning rate
optimizer = tf.keras.optimizers.Adamax(learning_rate=.25)
model = Model(inputData, N_communities, N_iterations, optimizer)
'''
# Adagrad doesn't converge below half of the SES value
# Nadam works but has shocky convergence
# Adam results in a very uniform map
# Adamax works well ->  lr=1 fails. 
#                       lr=.1 might be too uniform
#                       lr=.01 might be better but handling of weights and reg should be better

#TODO discuss the figures I get. Compare with how good the reguralisation works and use different regs
'''

print("INITIAL VALUES: ")
model.print_summary()

#%% Train the model for Niterations iterations
print("OPTIMISING...")
for i in range(model.OptimizationData.N_iterations):
    loss_value = model.train_step() # Take one optimization step
    if i % 10 == 0:
        # Print current loss and individual cost components
        print("Step: {}, Loss: {}".format(i, loss_value.numpy()))
        model.OptimizationData.printCosts()

model.applyMapCommunities()
print("FINISHED!\n")


#%% Plot the population on a map of Amsterdam    
img = plt.imread("Data/amsterdam.PNG")
fig, ax = plt.subplots()
#ax.imshow(img, extent=[-3000, 4500, -2300, 2000])
ax.imshow(img, extent=[-3000, 4000, -2300, 3000])
ax.scatter(model.InputData.Locations[:, 0], model.InputData.Locations[:, 1], s=model.InputData.Population/100, alpha=1, label="Neighbourhoods") # Plot locations
ax.scatter(model.Communities.Locations[:, 0], model.Communities.Locations[:, 1], s=model.Communities.Population/100, 
           alpha=1, label="Communities") # Plot community locations
ax.scatter(0, 0, alpha=.7) # Plot origin
ax.legend()


# Pltot 
fig1, ax1 = model.OptimizationData.plotCosts()


# histogram of the economic data 
fig2, ax2 = plt.subplots()
num_bins = model.InputData.Socioeconomic_data.shape[0]
SES_append = np.append(model.InputData.Socioeconomic_data, model.Communities.Socioeconomic_data)
initial_population_map = tf.round(model.InputData.Population[:,0] * model.initial_Map)
SES_initial = tf.matmul(initial_population_map, model.InputData.Socioeconomic_data)[:,0] / tf.reduce_sum(initial_population_map, axis=1)
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)

n, bins, patches = ax2.hist(model.InputData.Socioeconomic_data.numpy(), bins=bin_edges, color = 'r',edgecolor = "black",
                            alpha=0.3, label='Initial SES neighbourhoods', density=True)
n, bins, patches = ax2.hist(SES_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.4, label='Initial SES communities', density=True)
n, bins, patches = ax2.hist(model.Communities.Socioeconomic_data.numpy(), bins=bin_edges-bin_edges[0]/2, color = 'g',
                            alpha=0.5, label='Mapped SES', density=True)

ax2.legend(loc='upper right')
ax2.set_xlabel('SES')
ax2.set_ylabel('Frequency')
ax2.set_title('Comparison of the economic data by population')
plt.show()

print("OPTIMISED VALUES: ")
model.print_summary()
