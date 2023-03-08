# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:41:17 2023

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
from dataLoader import DataLoader



#%% Load data
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# Download the file named "CSV met statistische symbolen"
data = DataLoader("Data/SES_WOA_scores_per_wijk_en_buurt_08032023_175111.csv")

# Source: https://www.atlasleefomgeving.nl/kaarten
data.load_geo_data('Data/wijkenbuurten_2022_v1.GPKG')

data.buurt_filter()

#%% Translate locations to a grid
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode("Amsterdam").latitude , geolocator.geocode("Amsterdam").longitude ]


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