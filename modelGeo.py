# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:48:13 2023

@author: Mels
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)  ### TODO should be False
plt.close('all')

from inputData import InputData
from communities import Communities
from optimizationData import OptimizationData
import random

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
        
        if self.InputData.Locations is None:
            Exception('DataLoader has not initialised neighbourhood locations yet. Use the function DataLoader.map2grid(latlon0) to do this!')
        
        
        # Initialize communities
        self.Communities = Communities(N_communities)
        self.Communities.initialize_community_Locations(self.Communities.N, self.InputData.Locations)
        
        # Initialize the distance matrix and the economic values
        self.initialize_distances()
        
        # we label the data with a number that corresponds to the closest community
        self.labels = tf.Variable(tf.argmin(self.distances, axis=0).numpy(), trainable=True, dtype=tf.int32)
        self.labels_initial = tf.Variable(tf.argmin(self.distances, axis=0).numpy(), trainable=False, dtype=tf.int32)
        
        # initialise weights
        self.OptimizationData = OptimizationData(weights=[10,1,1,1], N_iterations=N_iterations,
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
    def Map(self, inputs):
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
        if tf.squeeze(inputs).shape[0]!=self.labels.shape[0]: 
            raise Exception("inputs should have the same lenght as self.labels!")
        indices = tf.transpose(tf.stack([self.labels, tf.range(self.InputData.N)]))
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
        #self.labels.assign(tf.math.mod(self.labels, self.Communities.N))
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
    
        # Regularization term to ensure population map is positive
        cost_popPositive = (tf.reduce_sum(tf.abs(tf.where(self.labels < 0, 1., 0.)))*100)
    
        # Regularization term for population limits
        cost_popBounds = tf.reduce_sum(tf.where(self.mapped_Population > self.OptimizationData.popBoundHigh,
                                              tf.abs(self.mapped_Population-self.OptimizationData.popBoundHigh), 0) +
                                     tf.where(self.mapped_Population < self.OptimizationData.popBoundLow, 
                                              tf.abs(self.mapped_Population-self.OptimizationData.popBoundLow), 0)) / self.tot_pop 
    
        # Add regularization term based on distances
        pop_distances = tf.multiply(self.population_Map, self.distances)
        cost_distance = tf.reduce_sum(pop_distances / self.tot_pop / self.max_distance)
        
        # Input costs into the OptimizationData model and save them
        self.OptimizationData.saveCosts(SES_variance, cost_popPositive, cost_popBounds, cost_distance)
        return self.OptimizationData.totalCost  
    
    
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
                
            if iteration % 10 == 0:
                # Print current loss and individual cost components
                print("Step: {}, Loss: {}".format(iteration, current_cost.numpy()))
                self.OptimizationData.printCosts()
                
            # Reduce the temperature at the end of each epoch
            temperature *= 0.99
            self.OptimizationData.storeCosts() # store the costs to be used later in a plot
            
    
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
        Locations: tf.float32 Tensor
            A tensor of shape (Communities.N, 2) containing the grid locations of the initial neighborhoods.
        Communities.Locations: tf.float32 Tensor
            A tensor of shape (InputData.N, 2) containing the grid locations of the newly created communities.

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
    
    
    @tf.function
    def print_summary(self):
        print(
          "\nThe Labels are:\n",self.labels.numpy(),
          "\n\nThe Population Map is:\n",tf.round( self.population_Map.numpy()),
          "\n\nSocioeconomic_data:\n", tf.expand_dims(self.mapped_Socioeconomic_data, axis=1).numpy(),
          "\n\nPopulation Size:\n", tf.round( tf.expand_dims(self.mapped_Population, axis=1) ).numpy()
          ,"\n\n")