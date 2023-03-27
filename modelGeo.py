# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:48:13 2023

@author: Mels
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import copy 

tf.config.run_functions_eagerly(True)  ### TODO should be False
plt.close('all')

from inputData import InputData
from communities import Communities
from optimizationData import OptimizationData

class ModelGeo(InputData, tf.keras.Model):
    def __init__(self, InputData, N_communities, N_iterations, optimizer):
        """
        Initializes the Dataset object with given socioeconomic data, population size, number of communities, and neighbourhood locations.
    
        Args:
            InputData (InputData Class): 
                (InputData.N x N_features) array containing the socioeconomic data of the initial neighbourhoods.
            N_communities (int): 
                The number of communities to end up with.
            N_iterations (ubt):
                The number of iterations of the optimizer
            optimizer (TensorFlow optimizer):
                The optimizer that will be used by the optimization
        
        Raises:
            Exception: If the number of new communities is greater than the number of neighbourhoods.
        
        Attributes:
            InputData.Socioeconomic_data (TensorFlow):
                (InputData.N x 1 x 1) TensorFlow variable containing the socioeconomic data of the initial neighbourhoods.
            InputData.Population (TensorFlow):
                (InputData.N x 1 x 1) TensorFlow variable containing the population sizes of the initial neighbourhoods.
            InputData.Locations (TensorFlow):
                (InputData.N x 2) TensorFlow variable containing the grid locations of the initial neighbourhoods.
            Communities.N (int):
                The number of communities to end up with.
            InputData.N (int):
                The number of initial neighbourhoods.
            Map (Tensor):
                (Communities.N x InputData.N) TensorFlow variable representing the community map.
            Communities.Locations (Variable):
                (Communities.N x 2) TensorFlow variable representing the center points of the new communities.
            population_bounds (Variable):
                (2,) TensorFlow variable representing the upper and lower boundaries by which the population can grow or shrink of their original size.
            tot_pop (Tensor):
               TensorFlow tensor representing the total population size of all initial neighbourhoods.
            avg_pop (Tensor):
                TensorFlow tensor representing the average population size of the initial neighbourhoods.
            popBoundHigh (Tensor):
                TensorFlow tensor representing the upper population boundary for the new communities.
            popBoundLow (Tensor):
                TensorFlow tensor representing the lower population boundary for the new communities
        """
        #super(Dataset, self).__init__()
        tf.keras.Model.__init__(self)
        
        # dataset should have the next variables: 
        #    neighbourhoods (numpy.ndarray): A 1D array of strings containing the names of the neighbourhoods.
        #    Population (numpy.ndarray): A 1D array of floats containing the number of private Population in each neighbourhood.
        #    Socioeconomic_data (numpy.ndarray): A 1D array of floats containing the socio-economic value of each neighbourhood.
        #    Locations (numpy.ndarray): A numpy array of shape (n,2) containing the mapped coordinates.
        self.InputData = InputData
        self.GeometryNeighbours = None
        
        if self.InputData.Locations is None:
            Exception('DataLoader has not initialised neighbourhood locations yet. Use the function DataLoader.map2grid(latlon0) to do this!')
        
        
        # Initialize communities
        self.Communities = Communities(N_communities)
        self.Communities.initialize_community_Locations(self.Communities.N, self.InputData.Locations)
        
        # Initialize the distance matrix and the economic values
        self.initialize_distances()
        self.initialize_labels()
        
        # initialise weights
        self.OptimizationData = OptimizationData(weights=[10,1,5,10], N_iterations=N_iterations,
                                                 LN=[1,2,1,1], optimizer=optimizer)
        
        # Initialize population parameters
        self.tot_pop = tf.reduce_sum(self.InputData.Population)
        self.avg_pop = self.tot_pop / self.Communities.N # Average population size
        self.OptimizationData.initialize_popBoundaries(self.avg_pop, population_bounds=[0.8, 1.2])
        
        self.initialize_weights()   
    '''    
    @property
    def labels(self):
        # define labels as a modulus
        return tf.math.mod(self.labels_var, self.Communities.N)
    '''
    
    @property
    def mapped_Population(self):
        return self(self.InputData.Population)
    
    @property
    def mapped_Socioeconomic_data(self):
        return self(self.InputData.Socioeconomic_population)/self.mapped_Population
    
    @property
    def population_Map(self):
        return tf.round(self.Map(self.InputData.Population))
    
    
    @tf.function
    def applyMapCommunities(self):
        self.Communities.Population = self.mapped_Population
        self.Communities.Socioeconomic_data = self.mapped_Socioeconomic_data
        
        
    @tf.function
    def Map(self, inputs, labels=None):
        '''
        Creates a Map of the inputs according to the labels

        Parameters
        ----------
        inputs : (InputData.N) or (InputData.N x 2) Tensor
            The inputs that we want to be transformed into a Map Tensor.

        Raises
        ------
        Exception
            If the length of inputs and labels are not equal.

        Returns
        -------
        (Communities.N x InputData.N) or (Communities.N x InputData.N x 2) Tensor
            The Mapped output
        '''
        if labels is None:
            if tf.squeeze(inputs).shape[0]!=self.labels.shape[0]: 
                raise Exception("inputs should have the same lenght as self.labels!")
            indices = tf.transpose(tf.stack([self.labels, tf.range(self.InputData.N)]))
            if len(tf.squeeze(inputs).shape)==2:
                return tf.scatter_nd(indices, tf.squeeze(inputs), shape = (self.Communities.N, self.InputData.N, 2))
            else:
                return tf.scatter_nd(indices, tf.squeeze(inputs), shape = (self.Communities.N, self.InputData.N))
            
        else: # in the case we give a different map        
            if tf.squeeze(inputs).shape[0]!=labels.shape[0]: 
                raise Exception("inputs should have the same lenght as self.labels!")
            indices = tf.transpose(tf.stack([labels, tf.range(self.InputData.N)]))
            if len(tf.squeeze(inputs).shape)==2:
                return tf.scatter_nd(indices, tf.squeeze(inputs), shape = (self.Communities.N, self.InputData.N, 2))
            else:
                return tf.scatter_nd(indices, tf.squeeze(inputs), shape = (self.Communities.N, self.InputData.N))
        
    
    @tf.function
    def call(self, inputs):
        '''
        Transforms the inputs according to the label

        Parameters
        ----------
        inputs : (InputData.N) or (InputData.N x 2) Tensor
            The inputs that we want to be transformed.

        Returns
        -------
        (Communities.N) or (Communities.N x 2) Tensor
            The transformed values of the Tensor.

        '''
        return tf.reduce_sum(self.Map(inputs), axis=1)
            
    
    @tf.function
    def cost_fn(self):
        """
        Calculates the cost function that needs to be minimized during the optimization process. The cost function consists of several
        regularization terms that encourage the population map to have certain properties, such as positive values and limits on population
        growth.
        
        Returns:
        tf.float32: A float representing the sum of all partial costs.
        
        Raises:
        ValueError: If the shape of any TensorFlow tensor used in the cost function is not as expected.
        
        Attributes:
        OptimizationData.weight_SESvariance (float):
        The weight given to the variance of the socioeconomic data mapped to the population map.
        OptimizationData.weight_popPositive (float):
        The weight given to the regularization term that ensures the population map is positive.
        OptimizationData.weight_popBounds (float):
        The weight given to the regularization term that limits the growth or shrinkage of population within a certain range.
        tot_pop (Tensor):
        A TensorFlow tensor representing the total population size of all initial neighbourhoods.
        max_distance (float):
        The maximum distance between two communities in the community map.
        OptimizationData.weight_distance (float):
        The weight given to the regularization term that penalizes communities that are too far apart.
        
        Returns:
        Tensor: A TensorFlow tensor representing the sum of all partial costs.
        """        
        # Calculate variance of socioeconomic data mapped to population map
        SES_variance = tf.math.reduce_variance( self(self.InputData.Socioeconomic_population) )
    
        # Regularization term for population limits
        cost_popBounds = tf.reduce_sum(tf.where(self.mapped_Population > self.OptimizationData.popBoundHigh,
                                              tf.abs(self.mapped_Population-self.OptimizationData.popBoundHigh), 0) +
                                     tf.where(self.mapped_Population < self.OptimizationData.popBoundLow, 
                                              tf.abs(self.mapped_Population-self.OptimizationData.popBoundLow), 0)) / self.tot_pop 
    
        # Add regularization term based on distances
        #pop_distances = tf.multiply(self.population_Map, self.distances)
        #cost_distance = tf.reduce_sum(pop_distances / self.tot_pop / self.max_distance)
        
        # Input costs into the OptimizationData model and save them
        self.OptimizationData.saveCosts(SES_variance, tf.Variable(0.), cost_popBounds, tf.Variable(0.))
        return self.OptimizationData.totalCost
    
    '''
    @tf.function
    def train(self, temperature=1.2):
        # Determine the initial cost of the current label assignment
        current_cost = self.cost_fn()
        self.OptimizationData.storeCosts() # store the costs to be used later in a plot
        
        # Iterate for a fixed number of epochs
        for iteration in range(self.OptimizationData.N_iterations):
            # Create a shuffled list of indices
            indices = list(range(self.labels.shape[0]))
            random.shuffle(indices)
            
            # Iterate over the shuffled indices
            for i in indices:
                # Try changing the label value for this element
                new_label = random.randrange(self.Communities.N)
                old_labels = tf.identity(self.labels).numpy()
                new_labels = tf.identity(self.labels).numpy()
                
                #if new_label==old_labels[i]: new_label = new_label+1%self.Communities.N
                new_labels[i] = new_label
                
                # Calculate the new cost and determine whether to accept the new label
                self.labels.assign(new_labels)
                new_cost = self.cost_fn()
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or random.uniform(0, 1) < tf.exp(-delta_cost / temperature):
                    current_cost = new_cost
                else:
                    self.labels.assign(old_labels)
                
            # Reduce the temperature at the end of each epoch
            temperature *= 0.99
            self.OptimizationData.storeCosts() # store the costs to be used later in a plot
    '''        
            
    @tf.function
    def refine(self, Nit, temperature=1.2):
        
        # load neigbhours
        if self.GeometryNeighbours is not None: 
            self.GeometryNeighbours = self.InputData.find_polygon_neighbors()
            
        for i in range(Nit):
            # iterate over the neighbourhoods 
            labelslist = list(range(self.InputData.N))
            random.shuffle(labelslist)
            for label in labelslist:
                label_old = self.labels[label]
                label_cost = self.cost_fn()
                
                # iterate over the neighbours to 
                neighbours = self.GeometryNeighbours[label]
                random.shuffle(neighbours)
                labellist_tried=[] # list of labels that have been tried already
                for neighbour in neighbours:
                    # check if label is not the same or hasn't been tried before
                    if ( self.labels[neighbour] != self.labels[label] and
                        self.labels[neighbour] not in labellist_tried):
                        
                        self.labels[label].assign(self.labels[neighbour])
                        labellist_tried.append(self.labels[label].numpy())
                        neighbour_cost = self.cost_fn()
                        
                        # choose to accept or reject
                        if neighbour_cost < label_cost:
                            label_cost = neighbour_cost
                                                        
                        else:
                            # Calculate the probability of accepting a higher-cost move
                            delta = neighbour_cost - label_cost
                            prob_accept = tf.exp(-delta / temperature)
                            # Accept the move with probability prob_accept
                            if random.uniform(0, 1) < prob_accept:
                                label_cost = neighbour_cost
                            else:
                                self.labels[label].assign(label_old)
                                
                                
            self.OptimizationData.storeCosts()    
            if i % 10 == 0:
                # Print current loss and individual cost components
                print("Step: {}, Loss: {}".format(i, label_cost.numpy()))
                self.OptimizationData.printCosts()
                
            temperature *= 0.99
        return self.labels
         
        
    '''  
    @tf.function
    def refine(self, Nit, Temperature):
        
        if self.GeometryNeighbours is not None: 
            self.GeometryNeighbours = self.InputData.find_polygon_neighbors()
            
        # create the list of neighbours for a label
        com_Neighbours = []
        labels = self.labels.numpy()
        for i in range(self.Communities.N):
            nghbrs = np.concatenate( self.GeometryNeighbours[ self.labels==i ] )
            
            # delete double indices or indices that are in the same group
            for index in range(len(nghbrs)-1,-1,-1):
                    if (nghbrs.count(nghbrs[index])>1 or 
                        nghbrs[index] in np.argwhere(self.labels==i) ):
                        nghbrs.pop(index)           
                        
            com_Neighbours.append(nghbrs)
            
        
        for i in range(Nit):
            # iterate over the communities and load their current values
            labels = self.labels.numpy()
            labelslist = list(range(self.Communities.N))
            random.shuffle(labelslist)
            for label in labelslist:
                label_cost = self.cost_fn()
                
                # iterate overthe neigbhours and change a label to see if it improves
                options = []
                for neighbour in com_Neighbours[label]:                    
                    labels[neighbour] = label
                    neighbour_cost = self.cost_fn(
                        tf.reduce_sum(self.Map(inputs), axis=1)
                        )

                    # choose which one move
                    if neighbour_cost < label_cost:
                        label_cost = neighbour_cost
                    else:
                        # Calculate the probability of accepting a higher-cost move
                        delta = neighbour_cost - label_cost
                        prob_accept = tf.exp(-delta / Temperature)
                        # Accept the move with probability prob_accept
                        if random.uniform(0, 1) < prob_accept:
                            label_cost = neighbour_cost
                        else:
                            self.labels[label] = label
        return self.labels
    '''
            
    
    @tf.function
    def initialize_weights(self):
        # Normalizes the weights such that relatively all costs start at 1. 
        # Then it multplies the normalized weights by the assigned weights
        self.OptimizationData.weight_SESvariance = self.OptimizationData.weight_SESvariance / ( 
            tf.math.reduce_variance( self(self.InputData.Socioeconomic_population) )
            )
        self.OptimizationData.weight_distance = self.OptimizationData.weight_distance / ( 
            tf.reduce_sum(tf.multiply(self.population_Map, self.distances) / self.tot_pop / self.max_distance)
            )  
        
        
    @tf.function
    def initialize_distances(self):
        """
        Initializes the pairwise distances between the newly created communities and the initial neighborhoods.

        Parameters:
        -----------
        InputData.Locations: tf.float32 Tensor
            A tensor of shape (InputData.N, 2) containing the grid locations of the initial neighborhoods.
        Communities.Locations: tf.float32 Tensor
            A tensor of shape (Communities.N, 2) containing the grid locations of the newly created communities.

        Returns:
        --------
        distances: tf.float32 Tensor
            A tensor of shape (InputData.N, Communities.N) containing the differences in distance between all indices.
        max_distance: tf.float32 scalar
            The maximum distance between any two locations.
        
        Raises:
        -------
        ValueError: If the shape of Locations or Communities.Locations is not as expected.
        """
        # Repeat the rows of Locations M times and the rows of Communities.Locations N times
        InputLocations_repeated = tf.repeat(tf.expand_dims(self.InputData.Locations, axis=0), self.Communities.N, axis=0)
        CommunitiesLocations_repeated = tf.tile(tf.expand_dims(self.Communities.Locations, axis=1), [1, self.InputData.N, 1])
    
        # Calculate the pairwise distances between all pairs of locations using the Euclidean distance formula
        self.distances = tf.sqrt(tf.reduce_sum(tf.square(InputLocations_repeated - CommunitiesLocations_repeated), axis=-1))
        self.max_distance = tf.reduce_max(self.distances)
        return self.distances
    
    
    def initialize_labels(self):
        '''
        Initialised the labels via:
            1. The neighbourhoods closest to the community centers will be initialised
                as those communities. The rest will be initiated with a Noe Label
            2. The model iterates over the communities and the adjecent neighbours of 
                those communities
                2.1. The model calculates which new neighbour (that is not part of another community)
                    would result in the SES value getting closer to the average SES
                2.2. The model chooses That neighbour and adds its values to the communities
                2.3. The model deletes neigbours of communities that are already part of said commynity
                2.4. The model evaluates if there are any neighbourhoods that are not part of 
                    Communities
                2.5. If not, the model stops. Else it will iterate further

        Raises
        ------
        Exception
            If the distances have not been initialised.

        Returns
        -------
        tf.Variable int  
            The newly calculated labels that map the neighbourhoods to their Communities.

        '''
        try: self.distances
        except: raise Exception("Distances should be initialized before running this function!")
        
        labels = [None for _ in range(self.InputData.N)]
        self.GeometryNeighbours = self.InputData.find_polygon_neighbors()
        
        # create the basis of the communities. The initial neighbours from which the communities spread out
        index = tf.argmin(self.distances, axis=1)
        com_Neighbours=[]       # the neighbours of the communities in the current state
        com_SES=[]              # the SES value of the communities in the current state
        com_index=[]            # the indices of neighbourhoods in the current communities
        for i in range(self.Communities.N):
            labels[index[i]]=i
            com_Neighbours.append(list(self.GeometryNeighbours[index[i]]))
            com_SES.append(self.InputData.Socioeconomic_population[index[i]].numpy())
            com_index.append([index[i].numpy()])        
                    
        # let the communities spread out to nearest neighbours
        avg_SES = tf.reduce_mean(self.InputData.Socioeconomic_population)
        while True:
            
            # iterate over the communities and load their current values
            Coms = list(range(self.Communities.N))
            random.shuffle(Coms)
            for i in Coms:
                index_current = com_index[i]
                
                # calculate the SES values of of adding the neighbourhoods to the list 
                SES_current = tf.reduce_sum(tf.gather(self.InputData.Socioeconomic_population, index_current))
                SES_neighbour = tf.gather(self.InputData.Socioeconomic_population, com_Neighbours[i])
                SES_option=(np.abs((SES_current + SES_neighbour - avg_SES).numpy()))
                    
                # choose the SES value that lies closest to the average
                ##TODO make sure this is done after the previous loop is done, such that it can decide double neighbourhoods
                for j in range(len(SES_option)): 
                    index_decision = com_Neighbours[i][np.argsort(SES_option)[j]]
                    if labels[index_decision] is None: # Neighbourhood should not be part of community already
                        break
                    
                if index_decision in com_index: raise Exception("The index_decision is already part of the community!")
                
                if labels[index_decision] is None: # in case all neighbours are already taken
                    # add the newly decided values to the community
                    labels[index_decision] = i
                    com_Neighbours[i] += list(self.GeometryNeighbours[index_decision])
                    com_SES[i] +=  tf.gather(self.InputData.Socioeconomic_population, index_decision).numpy()
                    com_index[i].append(index_decision)
                    
                # delete neighbours that are within the community
                for nghb_i in range(len(com_Neighbours[i])-1, -1, -1):
                    if com_Neighbours[i][nghb_i] in com_index:
                        com_Neighbours[i].pop(nghb_i)
                    #if com_Neighbours[i].count(com_Neighbours[i][nghb_i])>1:
                    #    com_Neighbours[i].pop(nghb_i)
                        
                            
            count = 0
            for label in labels:
                if label is None:
                    count+=1
            if count == 0: break
        
        
        self.labels = tf.Variable(labels)   
        self.applyMapCommunities()         
        return self.labels
    
    
    @tf.function
    def print_summary(self):
        print(
          "\nThe Labels are:\n",self.labels.numpy(),
          "\n\nThe Population Map is:\n",tf.round( self.population_Map.numpy()),
          "\n\nSocioeconomic_data:\n", tf.expand_dims(self.mapped_Socioeconomic_data, axis=1).numpy(),
          "\n\nPopulation Size:\n", tf.round( tf.expand_dims(self.mapped_Population, axis=1) ).numpy()
          ,"\n\n")