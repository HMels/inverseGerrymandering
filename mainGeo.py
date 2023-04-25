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
# trying to get 10000 households per community
N_communities = 10#int(np.sum(inputData.Population)/10000) # 20 # Number of communities
N_iterations = 100 # Number of iterations for training

# Define optimization algorithm and learning rate
optimizer = tf.keras.optimizers.Adamax(learning_rate=.05)
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
Population_initial = model.mapped_Population.numpy()
SES_initial = model.mapped_Socioeconomic_data.numpy()
Education_initial = model.mapped_Education.numpy()

print("INITIAL VALUES: ")
model.print_summary()

cdict = create_color_dict(N_communities)
fig01, ax01 = model.plot_communities(cdict, title='Communities Before Refinement')
    

#%% Train the model for Niterations iterations
print("OPTIMISING...")
model.refine(N_iterations, temperature=.05)
model.applyMapCommunities()
print("FINISHED!\n")


#%% Polygon Plot 
fig02, ax02 = model.plot_communities(cdict, title='Communities After Refinement')

#% Plotting
fig1, ax1 = model.OptimizationData.plotCosts()

print("OPTIMISED VALUES: ")
model.print_summary()

# factor by which to normalize the number of bins
normalization=3.5


#%% plot the SES value
fig2, ax2 = plt.subplots(figsize=(7.5, 4))
num_bins = int(model.InputData.Socioeconomic_data.shape[0]/normalization)
SES_append = np.append(model.InputData.Socioeconomic_data, model.Communities.Socioeconomic_data)
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)

# Plotting the barplot
n, bins, patches = ax2.hist(model.InputData.Socioeconomic_data.numpy(), bins=bin_edges, color = 'r',edgecolor = "grey",
                            alpha=0.3, label='Initial Neighbourhoods', density=True)
n, bins, patches = ax2.hist(SES_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.3, label='Initial Communities', density=True)
n, bins, patches = ax2.hist(model.Communities.Socioeconomic_data.numpy(), bins=bin_edges, color = 'g',
                            alpha=0.5, label='Refined Communities', density=True)

# turn of the axis that we don't need
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.autoscale_view('tight')
ax2.get_yaxis().set_visible(False)

# setting labels
ax2.legend(loc='upper right')
ax2.set_xlabel('Socio-economic score')
ax2.get_yaxis().set_visible(False)
ax2.set_title('Distribution of the economic data by population')
plt.show()


#%% histogram of the education
fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4))
num_bins = int(model.InputData.Education.shape[0]/normalization)
edu_append = np.append(model.InputData.Education, model.mapped_Education, axis=0)
bin_edges1 = np.linspace(np.min(edu_append[:,0]), np.max(edu_append[:,0]), num_bins+1)
bin_edges2 = np.linspace(np.min(edu_append[:,1]), np.max(edu_append[:,1]), num_bins+1)
bin_edges3 = np.linspace(np.min(edu_append[:,2]), np.max(edu_append[:,2]), num_bins+1)
bin_edges=[bin_edges1, bin_edges2, bin_edges3]

# plot bar plots for each column of InputData.Education
for i in range(3):
    n, bins, patches = ax3[i].hist(model.InputData.Education[:, i], bins=bin_edges[i], color='r', edgecolor='grey',
                                   alpha=.3, label='Initial Neighbourhoods', density=True)
    n, bins, patches = ax3[i].hist(Education_initial[:, i], bins=bin_edges[i], color='orange', edgecolor='grey',
                                   alpha=0.5, label='Initial Communities', density=True)
    n, bins, patches = ax3[i].hist(model.mapped_Education[:, i], bins=bin_edges[i], color='g',
                                   alpha=0.5, label='Refined Communities', density=True)

    # turn of the axis that we don't need
    ax3[i].spines['top'].set_visible(False)
    ax3[i].spines['right'].set_visible(False)
    ax3[i].get_yaxis().set_visible(False)
    ax3[i].autoscale_view('tight')

# set overall title for the figure
ax3[2].legend(loc='upper right')
ax3[1].set_xlabel('Percentage of the Population in Communities')
ax3[0].set_title('Low Level')
ax3[1].set_title('Medium Level')
ax3[2].set_title('High Level')
fig3.suptitle('Distribution of Educational Levels by Population')



#%% plot the Populations
fig4, ax4 = plt.subplots(figsize=(7.5, 4))
num_bins = int(model.InputData.Population.shape[0]/normalization)
Pop_appended = np.append(model.InputData.Population, model.mapped_Population)
bin_edges = np.linspace(np.min(Pop_appended), np.max(Pop_appended), num_bins+1)

# plot the bars
n, bins, patches = ax4.hist(model.InputData.Population.numpy(), bins=bin_edges, color = 'r',edgecolor = "grey",
                            alpha=0.3, label='Initial Neighbourhoods', density=True)
n, bins, patches = ax4.hist(Population_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.3, label='Initial Communities', density=True)
n, bins, patches = ax4.hist(model.mapped_Population.numpy(), bins=bin_edges, color = 'g',
                            alpha=0.5, label='Refined Communities', density=True)

# turn of the axis that we don't need
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.autoscale_view('tight')
ax4.get_yaxis().set_visible(False)

# labels
ax4.legend(loc='upper right')
ax4.set_xlabel('Population')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of population sizes')


#%% save all plots
if True:
    ax01.set_title('')
    ax02.set_title('')
    ax1.set_title('')
    ax2.set_title('')
    fig3.suptitle('')
    ax4.set_title('')
    #ax4.get_legend().remove()
    #ax3[2].get_legend().remove()
    
    fig01.tight_layout()
    fig02.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    
    fig01.savefig(fname="Output/01_CommunitiesBeforeRefinement")
    fig02.savefig(fname="Output/02_CommunitiesAfterRefinement")
    fig1.savefig(fname="Output/03_CostOtimizationPlot")

    fig2.savefig(fname="Output/04_SESbarplot")
    fig3.savefig(fname="Output/04_Educationbarplot")
    fig4.savefig(fname="Output/04_Populationbarplot")
