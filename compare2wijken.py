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

from modelGeo import ModelGeo


if __name__=="__main__":
    #% load wijken instead of buurten
    inputData_wijken = InputData("Data/SES_WOA_scores_per_wijk_en_buurt_25042023_160857.csv", False)
    
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
    normalization=5
    
    
    #%% calculate random average buurten
    with open("inputData.pickle", "rb") as f:
        inputData = pickle.load(f)
        
    N_communities= 10
    N_it = 50
    SES=[]
    Education0=[]
    Education1=[]
    Education2=[]
    Population=[]
    
    for i in range(N_it):
        print("Step:",str(i)+"/",str(N_it))
        model = ModelGeo(inputData, N_communities=N_communities)
        model.initialize_labels_random()
        
        SES.append(model.mapped_Socioeconomic_data.numpy())
        Education0.append(model.mapped_Education[:, 0].numpy())
        Education1.append(model.mapped_Education[:, 1].numpy())
        Education2.append(model.mapped_Education[:, 2].numpy())
        Population.append(model.mapped_Population.numpy())
        
        if i==0:
            fig01, ax01 = model.plot_communities(title='Communities Before Refinement')
        if i==1:
            fig02, ax02 = model.plot_communities(title='Communities Before Refinement')
        if i==2:
            fig03, ax03 = model.plot_communities(title='Communities Before Refinement')
    
    
    #%%
    SES = np.concatenate(SES)
    Education0 = np.concatenate(Education0)
    Education1 = np.concatenate(Education1)
    Education2 = np.concatenate(Education2)
    Population = np.concatenate(Population)
    
    #%% plot the SES value
    fig2, ax2 = plt.subplots(figsize=(7.5, 4))
    num_bins = int(SES_initial.shape[0]*normalization)
    SES_append = np.append(SES, mapped_SES)
    bin_edges = np.linspace(np.min(SES_append), np.max(SES_append), num_bins+1)
    
    # Plotting the barplot
    n, bins, patches = ax2.hist(SES, bins=bin_edges, color = 'r',edgecolor = "grey",
                                alpha=0.3, label='Random Neighbourhoods', density=True)
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
    ax2.set_xlabel('Average socio-economic score')
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Distribution of the mean socio-economic score')
    plt.show()
    
    
    #%% histogram of the education
    fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4))
    edu_append0 = np.append(Education0, mapped_Education[:,0], axis=0)
    edu_append1 = np.append(Education1, mapped_Education[:,1], axis=0)
    edu_append2 = np.append(Education2, mapped_Education[:,2], axis=0)
    bin_edges1 = np.linspace(np.min(edu_append0), np.max(edu_append0), num_bins+1)
    bin_edges2 = np.linspace(np.min(edu_append1), np.max(edu_append1), num_bins+1)
    bin_edges3 = np.linspace(np.min(edu_append2), np.max(edu_append2), num_bins+1)
    bin_edges=[bin_edges1, bin_edges2, bin_edges3]
    
    # plot bar plots for each column of inputData.Education
    n, bins, patches = ax3[0].hist(Education0, bins=bin_edges[0], color='r', edgecolor='grey',
                                   alpha=.3, label='Random Neighbourhoods', density=True)
    n, bins, patches = ax3[1].hist(Education1, bins=bin_edges[1], color='r', edgecolor='grey',
                                   alpha=.3, label='Random Neighbourhoods', density=True)
    n, bins, patches = ax3[2].hist(Education2, bins=bin_edges[2], color='r', edgecolor='grey',
                                   alpha=.3, label='Random Neighbourhoods', density=True)
    for i in range(3):
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
    Pop_appended = np.append(Population, mapped_Population)
    bin_edges = np.linspace(np.min(Pop_appended), np.max(Pop_appended), num_bins+1)
    
    # plot the bars
    n, bins, patches = ax4.hist(Population, bins=bin_edges, color = 'r',edgecolor = "grey",
                                alpha=0.3, label='Random Neighbourhoods', density=True)
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
        ax01.set_title('')
        ax02.set_title('')
        ax03.set_title('')
        ax2.set_title('')
        fig3.suptitle('')
        ax4.set_title('')
        try: ax4.get_legend().remove()
        except: pass
        try: ax3[2].get_legend().remove()
        except: pass
        
        fig01.tight_layout()
        fig02.tight_layout()
        fig03.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
    
        fig01.savefig(fname="SavedParameters/01_CommunitiesBeforeRefinement1")
        fig02.savefig(fname="SavedParameters/01_CommunitiesBeforeRefinement2")
        fig03.savefig(fname="SavedParameters/01_CommunitiesBeforeRefinement3")
        fig2.savefig(fname="SavedParameters/04_SESbarplot")
        fig3.savefig(fname="SavedParameters/04_Educationbarplot")
        fig4.savefig(fname="SavedParameters/04_Populationbarplot")
        
    
    #%%
    print("Variance: ",
          "\nSocioeconomic Score:",
          "\n Wijken:",np.var(inputData_wijken.Socioeconomic_data.numpy()),
          "\n Random Communities:",np.var(SES),
          "\n Initial Communities:",np.var(SES_initial),
          "\n Refined Communities:",np.var(mapped_SES),
          "\nEducation:",
          "\n Wijken:",np.var(inputData_wijken.Education.numpy(), axis=0),
          "\n Random Communities0:",np.var(Education0, axis=0),
          "\n Random Communities1:",np.var(Education1, axis=0),
          "\n Random Communities2:",np.var(Education2, axis=0),
          "\n Initial Communities:",np.var(Education_initial, axis=0),
          "\n Refined Communities:",np.var(mapped_Education, axis=0),
          "\nPopulation:",
          "\n Wijken:",np.var(inputData_wijken.Population.numpy(), axis=0),
          "\n Random Communities:",np.var(Population, axis=0)   ,
          "\n Initial Communities:",np.var(Population_initial),
          "\n Refined Communities:",np.var(mapped_Population) 
          )