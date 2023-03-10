# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:15 2023

@author: Mels

The file contains a Python class Model that inherits from two other classes, InputData 
and tf.keras.Model. It initializes a map of communities based on population and socio-economic data,
 and optimizes the placement of each community to minimize the cost of several constraints.

The class has several methods for mapping the input data to the community locations, calculating 
the cost function, and updating the placement of communities through gradient descent.

The Model class also contains several properties and functions for initializing weights and boundaries,
 as well as mapping population and socio-economic data to the community locations. 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)  ### TODO should be False
plt.close('all')

from inputData import InputData
from communities import Communities
from optimizationData import OptimizationData

class Model(InputData, tf.keras.Model):
    '''            
    Remarks: 
        We start with M=1 and just look at age
        By making M a linear mapping, we assume that all parameters in Data should 
            be linearly mapped. This might differ for more complex ratings.
        Because we don't want to iterate over all elements, we use a simplified version
            of entropy gotten from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
        It is probably best to make the M=0 vector of Data the population vector. 
            This would mean one could implement restrictions to the rows of Map
            multiplied with that vector of Data to restrict group sizes
        Right now entropy does not normalize data and thus each data has weird weights
        As the matrix is square, the amounts of neighbourhoods stays equal. 
        I should probably change it in such a way that the entropy is always created via map*population
        Geographical constraints should be more important that the amount of neighbourhoods. Assign
            coordinates to neighbourhoods and calculate distances in between
        For later version, we can limit the Map to a stroke in the diagonal to limit how far people can travel 
            between neighbourhoods
            
            
    Investigation into Entropy or KL-Divergence
        It seems like both should be used often in relation to a discrete dataset.
            So for example, the etnicity of people. Then the entropy would be able to
            calculate the probabilities of one belonging to a certain ethnicity being 
            put in a certain group and optimize that. Entropy works on probabilities
        Important to note, lim_x->0 x*ln(x) = 0 (lHopital). However, one would still
            need to map the Socioeconomic_data to fit in an interval of [0,1]
        I think it would be better to calculate the variance of the Socioeconomic_data
        
        
    Creating the communities
        In this case we totally assume the the geographical locations of the neighbourhoods
            are the centerpoints. This is false.
        Right now we use those locations to extrapolate where the locations of the new
            communities should be in initialize_community_Locations.
        This should ofcourse be better clarified. Center locations just do not represent
            what we are interested in.
            
    
    Calculating the distances
        We will work with a matrix that is representative of the distances between the
            Locations and the Communities.Locations 
        
    '''
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
        
        # Initialize the distance matrix and the map
        self.initialize_distances()        
        self.Map = self.initialize_map() # Community map
        
        # initialise weights
        self.OptimizationData = OptimizationData(weights=[10,1,1,3], N_iterations=N_iterations,
                                                 LN=[1,2,1,1], optimizer=optimizer)
        
        # Initialize population parameters
        self.tot_pop = tf.reduce_sum(self.InputData.Population)
        self.avg_pop = self.tot_pop / self.Communities.N # Average population size
        self.OptimizationData.initialize_popBoundaries(self.avg_pop, population_bounds=[0.8, 1.2])
        
        self.initialize_weights()        
        
        
    @property
    def mapped_Population(self):
        return self(self.InputData.Population)
    
    @property
    def mapped_Socioeconomic_data(self):
        return self(self.InputData.Socioeconomic_population)/self.mapped_Population
    
    @property
    def population_Map(self):
        return tf.round(self.InputData.Population * self.normalize_map())
    
    
    @tf.function
    def applyMapCommunities(self):
        self.Communities.Population = self.mapped_Population
        self.Communities.Socioeconomic_data = self.mapped_Socioeconomic_data
    
    
    @tf.function
    def call(self, inputs):
        '''Transforms the inputs according to the map'''
        self.Map.assign(self.normalize_map()) # Normalize the community map
        return tf.squeeze(tf.matmul(self.Map, tf.expand_dims(inputs, axis=1)))
    
    
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
        SES_variance = tf.math.reduce_variance( self(self.InputData.Socioeconomic_population) ) * self.OptimizationData.weight_SESvariance
    
        # Regularization term to ensure population map is positive
        cost_popPositive = (tf.reduce_sum(tf.abs(tf.where(self.Map < 0., self.Map, 0.))) * self.OptimizationData.weight_popPositive*100)
    
        # Regularization term for population limits
        cost_popBounds = tf.reduce_sum(tf.where(self.mapped_Population > self.OptimizationData.popBoundHigh,
                                              tf.abs(self.mapped_Population-self.OptimizationData.popBoundHigh), 0) +
                                     tf.where(self.mapped_Population < self.OptimizationData.popBoundLow, 
                                              tf.abs(self.mapped_Population-self.OptimizationData.popBoundLow), 0)) * ( 
                                         self.OptimizationData.weight_popBounds / self.tot_pop 
                                         )
    
        # Add regularization term based on distances
        pop_distances = tf.multiply(self.population_Map, self.distances)
        cost_distance = tf.reduce_sum(pop_distances / self.tot_pop / self.max_distance)*self.OptimizationData.weight_distance

        # Record the partial costs for inspection and return the sum of all partial costs
        self.OptimizationData.saveCosts(SES_variance, cost_popPositive, cost_popBounds, cost_distance)
        return self.OptimizationData.sumCosts
    
    
    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            loss_value = self.cost_fn()
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.OptimizationData.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value  
            
    
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
    def initialize_map(self):
        """
        Initializes the Map matrix that maps the data using a power third logarithmic distances. The map has the size 
        (final number of communities x initial number of communities). The map chooses how to spread around the population over
        the communities by using a power third logarithmic distances. So if two neighbourhoods are in the same order of 
        distance from said community, the map will spread the population over them equally. However, this spreading around 
        effect becomes strongly less as distance increases.
        
        Returns:
            A TensorFlow variable with shape (Communities.N, InputData.N), initialized with the desired values and set as trainable.            
        
        Attributes:
            self.distances (TensorFlow tensor):
                (Communities.N x InputData.N) TensorFlow tensor representing the distances between communities and initial neighborhoods.
            Communities.N (int):
                The number of communities to end up with.
            InputData.N (int):
                The number of initial neighborhoods.
        
        Returns:
            Map (TensorFlow variable):
                A TensorFlow variable with shape (Communities.N, InputData.N), initialized with the desired values and set as trainable.
        """

        # we use the third power to make the falloff steep for higher distances
        Map = tf.exp(-1*( self.distances / tf.reduce_min(self.distances, axis=0) )**3 )
        Map = np.round(Map, 1) # filter for the 10th percentiles
        Map = Map / tf.abs(tf.reduce_sum(Map, axis=0) ) # normalise 
        
        # Return the initialized Map matrix as a TensorFlow variable
        Map = tf.Variable(Map, dtype=tf.float32, trainable=True)
        self.initial_Map = tf.Variable(Map, dtype=tf.float32, trainable=False)
        return Map
    
    
    @tf.function 
    def normalize_map(self):
        '''
        Normalizes the map such that it does not create people or split people 
        over different communities.
        '''
        # Divide each row by its sum
        Map = self.Map / tf.abs(tf.reduce_sum(self.Map, axis=0) )
        return Map
    
    
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
          "\nThe Map is:\n",tf.round( self.normalize_map() * 100 , 1 ).numpy(),
          "\n\nwhich counts up to:\n",tf.reduce_sum(tf.round( self.normalize_map() * 100 , 1 ).numpy(), axis=0),
          "\n\nThe Population Map is:\n",tf.round( self.population_Map.numpy()),
          "\n\nSocioeconomic_data:\n", tf.expand_dims(self.mapped_Socioeconomic_data, axis=1).numpy(),
          "\n\nPopulation Size:\n", tf.round( tf.expand_dims(self.mapped_Population, axis=1) ).numpy()
          ,"\n\n")