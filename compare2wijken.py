# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:00:28 2023

@author: Mels
"""
import numpy as np
import pickle
from geopy.geocoders import Nominatim

from inputData import InputData
import matplotlib.pyplot as plt

#%% load wijken instead of buurten
inputData = InputData("Data/SES_WOA_scores_per_wijk_en_buurt_25042023_160857.csv", False)


#%% load model outputs
# initial
Population_initial=np.load('SavedParameters/Population_initial.npz')['my_array']
SES_initial=np.load('SavedParameters/SES_initial.npz')['my_array']
Education_initial=np.load('SavedParameters/Education_initial.npz')['my_array']
Distances_initial=np.load('SavedParameters/Distances_initial.npz')['my_array']

# eventual
mapped_Population=np.load('SavedParameters/Population.npz')['my_array']
mapped_SES=np.load('SavedParameters/SES.npz')['my_array']
mapped_Education=np.load('SavedParameters/Education.npz')['my_array']
Distances=np.load('SavedParameters/Distances.npz')['my_array']


# factor by which to normalize the number of bins
normalization=3.5

#%% plot the SES value
fig2, ax2 = plt.subplots(figsize=(7.5, 4))
num_bins = int(inputData.Socioeconomic_data.shape[0]/normalization)
SES_append = np.append(inputData.Socioeconomic_data, mapped_SES)
bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)

# Plotting the barplot
n, bins, patches = ax2.hist(inputData.Socioeconomic_data.numpy(), bins=bin_edges, color = 'r',edgecolor = "grey",
                            alpha=0.3, label='Initial Neighbourhoods', density=True)
n, bins, patches = ax2.hist(SES_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.3, label='Initial Communities', density=True)
n, bins, patches = ax2.hist(mapped_SES, bins=bin_edges, color = 'g',
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
num_bins = int(inputData.Education.shape[0]/normalization)
edu_append = np.append(inputData.Education, mapped_Education, axis=0)
bin_edges1 = np.linspace(np.min(edu_append[:,0]), np.max(edu_append[:,0]), num_bins+1)
bin_edges2 = np.linspace(np.min(edu_append[:,1]), np.max(edu_append[:,1]), num_bins+1)
bin_edges3 = np.linspace(np.min(edu_append[:,2]), np.max(edu_append[:,2]), num_bins+1)
bin_edges=[bin_edges1, bin_edges2, bin_edges3]

# plot bar plots for each column of inputData.Education
for i in range(3):
    n, bins, patches = ax3[i].hist(inputData.Education[:, i], bins=bin_edges[i], color='r', edgecolor='grey',
                                   alpha=.3, label='Initial Neighbourhoods', density=True)
    n, bins, patches = ax3[i].hist(Education_initial[:, i], bins=bin_edges[i], color='orange', edgecolor='grey',
                                   alpha=0.5, label='Initial Communities', density=True)
    n, bins, patches = ax3[i].hist(mapped_Education[:, i], bins=bin_edges[i], color='g',
                                   alpha=0.5, label='Refined Communities', density=True)

    # turn of the axis that we don't need
    ax3[i].spines['top'].set_visible(False)
    ax3[i].spines['right'].set_visible(False)
    ax3[i].get_yaxis().set_visible(False)
    ax3[i].autoscale_view('tight')

# set overall title for the figure
ax3[2].legend(loc='upper right')
ax3[1].set_xlabel('Percentage of the Population in Communities per Educational Level')
ax3[0].set_title('Low Level')
ax3[1].set_title('Medium Level')
ax3[2].set_title('High Level')
fig3.suptitle('Distribution of Educational Levels by Population')



#%% plot the Populations
fig4, ax4 = plt.subplots(figsize=(7.5, 4))
num_bins = int(inputData.Population.shape[0]/normalization)
Pop_appended = np.append(inputData.Population, mapped_Population)
bin_edges = np.linspace(np.min(Pop_appended), np.max(Pop_appended), num_bins+1)

# plot the bars
n, bins, patches = ax4.hist(inputData.Population.numpy(), bins=bin_edges, color = 'r',edgecolor = "grey",
                            alpha=0.3, label='Initial Neighbourhoods', density=True)
n, bins, patches = ax4.hist(Population_initial, bins=bin_edges, color = 'orange',edgecolor = "grey",
                                                        alpha=0.3, label='Initial Communities', density=True)
n, bins, patches = ax4.hist(mapped_Population, bins=bin_edges, color = 'g',
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


#%% save all plots and variables
if True:
    ax2.set_title('')
    fig3.suptitle('')
    ax4.set_title('')
    try: ax4.get_legend().remove()
    except: pass
    try: ax3[2].get_legend().remove()
    except: pass
    
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    fig2.savefig(fname="SavedParameters/04_SESbarplot")
    fig3.savefig(fname="SavedParameters/04_Educationbarplot")
    fig4.savefig(fname="SavedParameters/04_Populationbarplot")