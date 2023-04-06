# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:41:17 2023

@author: Mels
"""
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


#%% load inputData
with open("inputData.pickle", "rb") as f:
    inputData = pickle.load(f)


#%% Define model
N_communities = 5 # Number of communities
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


#%% plot initial state
SES_initial = model.mapped_Socioeconomic_data.numpy()
Education_initial = model.mapped_Education_population.numpy()

print("INITIAL VALUES: ")
model.print_summary()

#cdict = {0: 'c', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5:}
cdict = create_color_dict(N_communities)
extent=[-2000, 2000, -1200, 2800]
fig01, ax01 = model.plot_communities(extent, cdict, title='Communities Before Refinement')
    

#%% Train the model for Niterations iterations
print("OPTIMISING...")
model.refine(N_iterations, temperature=0)
model.applyMapCommunities()
print("FINISHED!\n")


#%% Polygon Plot 
fig02, ax02 = model.plot_communities(extent, cdict, title='Communities After Refinement')

#% Plotting
fig1, ax1 = model.OptimizationData.plotCosts()


############ histogram of the economic data 
fig2, ax2 = plt.subplots()
num_bins = model.InputData.Socioeconomic_data.shape[0]
SES_append = np.append(model.InputData.Socioeconomic_data, model.Communities.Socioeconomic_data)
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)


############ plot the SES value
n, bins, patches = ax2.hist(model.InputData.Socioeconomic_data.numpy(), bins=bin_edges, color = 'r',edgecolor = "black",
                            alpha=0.3, label='Initial Neighbourhoods', density=True)
n, bins, patches = ax2.hist(SES_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.3, label='Initial Communities', density=True)
n, bins, patches = ax2.hist(model.Communities.Socioeconomic_data.numpy(), bins=bin_edges-bin_edges[0]/2, color = 'g',
                            alpha=0.5, label='Refined Communities', density=True)

ax2.legend(loc='upper right')
ax2.set_xlabel('SES')
ax2.set_ylabel('Frequency')
ax2.set_title('Comparison of the economic data by population')
plt.show()


#%% histogram of the education
fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4))
num_bins = model.InputData.Education.shape[0]
edu_append = np.append(model.InputData.Education_population, model.mapped_Education_population)
bin_edges = np.linspace(min(edu_append), max(edu_append), num_bins+1)

# plot bar plots for each column of InputData.Education
for i in range(3):
    n, bins, patches = ax3[i].hist(model.InputData.Education_population[:, i], bins=bin_edges, color='r', edgecolor='black',
                                   alpha=.3, label='Initial Neighbourhoods', density=True)
    n, bins, patches = ax3[i].hist(Education_initial[:, i], bins=bin_edges, color='orange', edgecolor='grey',
                                   alpha=0.3, label='Initial Communities', density=True)
    n, bins, patches = ax3[i].hist(model.mapped_Education_population[:, i], bins=bin_edges-bin_edges[0]/2, color='g',
                                   alpha=0.5, label='Refined Communities', density=True)


# set overall title for the figure
ax3[2].legend(loc='upper right')
ax3[1].set_xlabel('Education Level')
ax3[0].set_ylabel('Frequency')
ax3[0].set_title('Low Level')
ax3[1].set_title('Medium Level')
ax3[2].set_title('High Level')
ax3[1].get_yaxis().set_visible(False)
ax3[2].get_yaxis().set_visible(False)
fig3.suptitle('Comparison of Educational Levels by Population')


print("OPTIMISED VALUES: ")
model.print_summary()


#%% save all plots
if True:
    fig01.savefig(fname="Output/01_CommunitiesBeforeRefinement")
    fig02.savefig(fname="Output/02_CommunitiesAfterRefinement")
    fig1.savefig(fname="Output/03_CostOtimizationPlot")

    fig2.savefig(fname="Output/04_SESbarplot")
    fig3.savefig(fname="Output/04_Educationbarplot")
