# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:15 2023

@author: Mels
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)  ### TODO should be False
plt.close('all')


class Dataset(tf.keras.Model):
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
            need to map the socioeconomic_data to fit in an interval of [0,1]
        I think it would be better to calculate the variance of the socioeconomic_data
        
        
    Creating the communities
        In this case we totally assume the the geographical locations of the neighbourhoods
            are the centerpoints. This is false.
        Right now we use those locations to extrapolate where the locations of the new
            communities should be in initialize_communities.
        This should ofcourse be better clarified. Center locations just do not represent
            what we are interested in.
            
    
    Calculating the distances
        We will work with a matrix that is representative of the distances between the
            neighbourhood_locs and the community_locs 
        
    '''
    def __init__(self, socioeconomic_data, Population_size, N_communities, neighbourhood_locs):
        """
        Initializes the Dataset object with given socioeconomic data, population size, number of communities, and neighbourhood locations.
    
        Parameters
        ----------
        socioeconomic_data : ndarray
            A (N_neighbourhoods x N_features) array containing the socioeconomic data of the initial neighbourhoods.
        population_size : ndarray
            A (N_neighbourhoods x 1) array containing the population sizes of the initial neighbourhoods.
        n_communities : int
            The desired number of communities to end up with.
        neighbourhood_locs : ndarray
            A (N_neighbourhoods x 2) array containing the grid locations of the initial neighbourhoods.
    
        Raises
        ------
        Exception
            If the number of new communities is greater than the number of neighborhoods.
    
        Attributes
        ----------
        socioeconomic_data : Variable
            A (N_neighbourhoods x 1 x N_features) TensorFlow variable containing the socioeconomic data of the initial neighbourhoods.
        population_size : Variable
            A (N_neighbourhoods x 1 x 1) TensorFlow variable containing the population sizes of the initial neighbourhoods.
        neighbourhood_locs : Variable
            A (N_neighbourhoods x 2) TensorFlow variable containing the grid locations of the initial neighbourhoods.
        N_communities : int
            The desired number of communities to end up with.
        N_neighbourhoods : int
            The number of initial neighbourhoods.
        Map : Variable
            A (N_neighbourhoods x 1 x N_communities) TensorFlow variable representing the community map.
        community_locs : Variable
            A (N_communities x 2) TensorFlow variable representing the center points of the new communities.
        population_bounds : Variable
            A (2,) TensorFlow variable representing the upper and lower boundaries by which the population can grow or shrink of their original size.
        tot_pop : Tensor
            A TensorFlow tensor representing the total population size of all initial neighbourhoods.
        avg_pop : Tensor
            A TensorFlow tensor representing the average population size of the initial neighbourhoods.
        popBoundHigh : Tensor
            A TensorFlow tensor representing the upper population boundary for the new communities.
        popBoundLow : Tensor
            A TensorFlow tensor representing the lower population boundary for the new communities.
        """
        super(Dataset, self).__init__()

        # Initialize inputs
        self.socioeconomic_data = tf.Variable(socioeconomic_data[:, None], trainable=False, dtype=tf.float32) # SES values
        self.population_size = tf.Variable(Population_size[:, None], trainable=False, dtype=tf.float32) # Population sizes
        self.neighbourhood_locs = tf.Variable(neighbourhood_locs, trainable=False, dtype=tf.float32) # Neighborhood locations
        
        # Initialize parameters
        self.N_communities = N_communities
        self.N_neighbourhoods = self.socioeconomic_data.shape[0]
        
        # Create the center points for the new communities
        if self.N_communities == self.N_neighbourhoods:
            # If the number of new communities is the same as the number of neighborhoods, use the same locations
            self.community_locs = tf.Variable(neighbourhood_locs, trainable=False, dtype=tf.float32)
        elif self.N_communities < self.N_neighbourhoods:
            # If the number of new communities is less than the number of neighborhoods, initialize new locations
            self.community_locs = self.initialize_communities(self.N_communities)
        else:
            # If the number of new communities is greater than the number of neighborhoods, raise an exception
            raise Exception("Model is not able to create more communities than were originally present!")
        
        # Initialize the distance matrix and the map
        self.initialize_distances()        
        self.Map = self.initialize_map() # Community map
        
        # Initialize population parameters
        self.population_bounds = tf.Variable([0.8, 1.2], trainable=False, dtype=tf.float32) # Boundaries by which the population can grow or shrink of their original size
        self.tot_pop = tf.reduce_sum(self.population_size)
        self.avg_pop = self.tot_pop / self.N_communities # Average population size
        self.popBoundHigh = self.population_bounds[1] * self.avg_pop # Upper population boundary
        self.popBoundLow = self.population_bounds[0] * self.avg_pop # Lower population boundary
        
        # Initialize the weights
        self.weight_SESvariance = 10
        self.weight_popPositive = 1
        self.weight_popBounds = 1
        self.weight_distance = 1
        
        self.initialize_weights()
        
    
    @property
    def mapped_population_size(self):
        return self(self.population_size)
    
    @property
    def mapped_socioeconomic_data(self):
        SES = tf.matmul(self.population_Map, self.socioeconomic_data)
        return SES/self.mapped_population_size
    
    @property
    def population_Map(self):
        return tf.round(self.population_size[:,0] * self.normalize_map())
    
    
    @tf.function
    def call(self, inputs):
        '''Transforms the inputs according to the map'''
        self.Map.assign(self.normalize_map()) # Normalize the community map
        return tf.matmul(self.Map, inputs)
    
    
    @tf.function
    def cost_fn(self):
        # Calculate variance of socioeconomic data mapped to population map
        SES_variance = tf.math.reduce_variance(tf.matmul(self.population_Map, self.socioeconomic_data) ) * self.weight_SESvariance
    
        # Regularization term to ensure population map is positive
        cost_popPositive = (tf.reduce_sum(tf.abs(tf.where(self.Map < 0., self.Map, 0.))) * self.weight_popPositive*100)
    
        # Regularization term for population limits
        cost_popBounds = tf.reduce_sum(tf.where(self.mapped_population_size > self.popBoundHigh,
                                              tf.abs(self.mapped_population_size-self.popBoundHigh), 0) +
                                     tf.where(self.mapped_population_size < self.popBoundLow, 
                                              tf.abs(self.mapped_population_size-self.popBoundLow), 0)) * ( 
                                         self.weight_popBounds / self.tot_pop 
                                         )
    
        # Add regularization term based on distances
        pop_distances = tf.multiply(self.population_Map, self.distances)
        cost_distance = tf.reduce_sum(pop_distances / self.tot_pop / self.max_distance)*self.weight_distance
                
        # Record the partial costs for inspection
        self.SES_variance = SES_variance
        self.L2_popPositive = cost_popPositive**2
        self.L1_popBounds = cost_popBounds
        self.L1_distance = cost_distance
    
        # Return the sum of all partial costs
        return self.SES_variance + self.L2_popPositive + self.L1_popBounds + self.L1_distance
    
    
    @tf.function
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:
            loss_value = self.cost_fn()
        grads = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value    
    
    
    @tf.function
    def initialize_weights(self):
        # Normalizes the weights such that relatively all costs start at 1. 
        # Then it multplies the normalized weights by the assigned weights
        self.weight_SESvariance = self.weight_SESvariance / ( 
            tf.math.reduce_variance(tf.matmul(self.population_Map, self.socioeconomic_data) )
            )
        self.weight_distance = self.weight_distance / ( 
            tf.reduce_sum(tf.multiply(self.population_Map, self.distances) / self.tot_pop / self.max_distance)
            )
    
    
    @tf.function
    def initialize_map(self):
        """
        Initialize the Map matrix that maps the data. The map has the size
        (final number of communities x initial number of communities). The map 
        chooses how to spread around the population over the communities by using 
        a power third logarithmic distances. So if two neighbourhoods are in the same 
        order of distance from said community, the map will spread the population 
        over them equally. However, this spreading around effect becomes strongly 
        less as distance increases.
    
        Returns:
            A TensorFlow variable with shape (N_communities, N_neighbourhoods), 
            initialized with the desired values and set as trainable.
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
    def initialize_communities(self, N_communities):
        """
        Initialize the locations of communities by sparsifying the input locations using KNN.
    
        Parameters
        ----------
        N_communities : int32
            The number of communities we want to end up with.
    
        Returns
        -------
        community_locs : float32 (N_communities x 2) array
            Array containing the grid locations of the newly created communities. 
        """
        # Define the number of nearest neighbors to consider
        k = tf.cast(tf.math.ceil(self.neighbourhood_locs.shape[0] / N_communities), tf.int32)

        # Calculate the Euclidean distances between all points in the data set
        distances = tf.reduce_sum(tf.square(tf.expand_dims(self.neighbourhood_locs, 1) - tf.expand_dims(self.neighbourhood_locs, 0)), axis=-1)

        # Find the indices of the nearest neighbors for each point
        _, nearest_neighbor_indices = tf.nn.top_k(-distances, k=k, sorted=True)
        
        # Gather the nearest neighbors for each point
        nearest_neighbor_locs = tf.gather(self.neighbourhood_locs, nearest_neighbor_indices, axis=0)

        # Reshape the nearest neighbors tensor into the desired shape
        nearest_neighbor_locs_reshaped = tf.reshape(nearest_neighbor_locs, [-1, k, 2])

        # Pick every M-th point from the new data set
        # M = k because we want to pick one community from each set of nearest neighbors
        sparse_indices = tf.round(tf.linspace(0,tf.shape(nearest_neighbor_locs_reshaped)[0]-1, N_communities))
        sparse_locs = tf.cast(tf.gather(nearest_neighbor_locs_reshaped, tf.cast(sparse_indices, tf.int32), axis=0), dtype=tf.float32)
        return tf.reduce_mean(sparse_locs, axis=1)
    
    
    @tf.function
    def initialize_distances(self):
        """
        Parameters
        ----------
        neighbourhood_locs : float32 (N_communities x 2) array
            Array containing the grid locations of the initial neighbourhoods 
        community_locs : float32 (N_neighbourhoods x 2) array
            Array containing the grid locations of the newly created communities 
    
        Returns
        -------
        distances : float32 (N_neighbourhoods x N_communities) array
            Array containing the differences in distance between all indices
        max_distance : float32 scalar
            Maximum distance between any two locations.
        
        This function initializes the pairwise distances between the newly created communities and the initial neighbourhoods.
        """
        # Repeat the rows of neighbourhood_locs M times and the rows of community_locs N times
        neighbourhood_locs_repeated = tf.repeat(tf.expand_dims(self.neighbourhood_locs, axis=0), self.N_communities, axis=0)
        community_locs_repeated = tf.tile(tf.expand_dims(self.community_locs, axis=1), [1, self.N_neighbourhoods, 1])
    
        # Calculate the pairwise distances between all pairs of locations using the Euclidean distance formula
        self.distances = tf.sqrt(tf.reduce_sum(tf.square(neighbourhood_locs_repeated - community_locs_repeated), axis=-1))
        self.max_distance = tf.reduce_max(self.distances)
        return self.distances
    
    
    @tf.function
    def print_summary(self):
        print(
          "\nThe Map is:\n",tf.round( self.normalize_map() * 100 , 1 ).numpy(),
          "\n\nwhich counts up to:\n",tf.reduce_sum(tf.round( self.normalize_map() * 100 , 1 ).numpy(), axis=0),
          "\n\nThe Population Map is:\n",tf.round( self.population_Map.numpy()),
          "\n\nsocioeconomic_data:\n", self.mapped_socioeconomic_data.numpy(),
          "\n\nPopulation Size:\n", tf.round( self.mapped_population_size ).numpy()
          ,"\n\n")