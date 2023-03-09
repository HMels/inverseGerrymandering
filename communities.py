# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:09:21 2023

@author: Mels

This is a Python file containing a class Communities and some helper functions.
 The class has an __init__ method to initialize the object, an initialize_community_Locations 
 method to initialize the locations of communities using K-means clustering, and some helper 
 functions such as KMeansClustering and FindSparseLocations for calculating the central points
 of communities. The file also imports numpy, tensorflow, and scikit-learn libraries to perform
 the clustering and distance calculations. 
"""

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class Communities:
    def __init__(self, N_communities, Population=None, Socioeconomic_data=None):
        """
        Initialize the communities object.
    
        Parameters
        ----------
        N_communities : int
            The number of communities to be created.
        Population : numpy.ndarray, optional
            An array containing population data for each community. The default is None.
        Socioeconomic_data : numpy.ndarray, optional
            An array containing socioeconomic data for each community. The default is None.
        """
        self.N = N_communities
        self.Population = Population if Population is not None else None
        self.Socioeconomic_data = Socioeconomic_data if Socioeconomic_data is not None else None
        
        
    @property
    def Socioeconomic_population(self):
        '''
        Returns
        -------
        tf.float32
            The socioeconomic data multiplied by the population to get the actual socioeconomic value.

        '''
        return self.Socioeconomic_data * self.Population
        
    
    @tf.function
    def initialize_community_Locations(self, N_communities, InputData_Locations):
        """
        Initialize the locations of communities by sparsifying the input locations using KNN.
    
        Parameters
        ----------
        N_communities : int
            The number of communities we want to end up with.
        N_inputData : int
            The number of input data points.
        InputData_Locations : numpy.ndarray
            An array containing the location data for the input data points.
    
        Returns
        -------
        Locations : numpy.ndarray
            Array containing the grid locations of the newly created communities.
        """
        N_inputData = InputData_Locations.shape[0]
        # Create the center points for the new communities
        if N_communities == N_inputData:
            # If the number of new communities is the same as the number of neighbourhoods, use the same locations
            self.Locations = tf.Variable(InputData_Locations, trainable=False, dtype=tf.float32)
        elif N_communities < N_inputData:
            # If the number of new communities is less than the number of neighbourhoods, initialize new locations
            self.Locations = tf.Variable(self.KMeansClustering(N_communities, InputData_Locations), trainable=False, dtype=tf.float32)
        else:
            # If the number of new communities is greater than the number of neighbourhoods, raise an exception
            raise Exception("Model is not able to create more communities than were originally present!")
            
            
    
    def KMeansClustering(self, N_communities, InputData_Locations):
        """
        Finds N_communities central points that are distributed over the data in such a way that all InputData_Locations 
        have a point that is close to them, while these points should not be too close to each other.

        Parameters
        ----------
        N_communities : int
            The number of communities we want to end up with.
        InputData_Locations : numpy.ndarray
            An (N_inputData x 2) array containing the location data for the input data points.

        Returns
        -------
        numpy.ndarray
            An (N_communities x 2) array containing the locations of the central points.
        """    
        # Step 1: Initialize the KMeans object and fit the data to it
        kmeans = KMeans(n_clusters=N_communities)
        kmeans.fit(InputData_Locations)

        # Step 2: Calculate the distances between each pair of centroids
        distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)

        # Step 3: Find the pair of centroids with the maximum distance
        i, j = np.unravel_index(distances.argmax(), distances.shape)

        # Step 4: Merge the two centroids and re-fit the data to the KMeans object
        kmeans.cluster_centers_[i] = np.mean(
            [kmeans.cluster_centers_[i], kmeans.cluster_centers_[j]], axis=0)
        kmeans.cluster_centers_ = np.delete(kmeans.cluster_centers_, j, axis=0)
        kmeans.fit(InputData_Locations)

        # Step 5: Repeat steps 2-4 until we have N_communities centroids
        while kmeans.n_clusters > N_communities:
            distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
            i, j = np.unravel_index(distances.argmax(), distances.shape)
            kmeans.cluster_centers_[i] = np.mean(
                [kmeans.cluster_centers_[i], kmeans.cluster_centers_[j]], axis=0)
            kmeans.cluster_centers_ = np.delete(kmeans.cluster_centers_, j, axis=0)
            kmeans.fit(InputData_Locations)

        # Step 6: Return the final centroids
        return kmeans.cluster_centers_
    
    
    def FindSparseLocations(N_communities, InputData_Locations):
        """
        This function uses a sparse sampling approach to find a set of N_communities central points that are well distributed across the InputData_Locations while satisfying a nearest neighbor condition based on the number of nearest neighbors k.
        Function Not in use Right Now
        
        Parameters
        ----------
        N_communities : int
            The number of communities we want to end up with.
        N_inputData : int
            The number of input data points.
        InputData_Locations : tensorflow.Tensor of shape (N_inputData, 2)
            A tensor containing the location data for the input data points.
    
        Returns
        -------
        sparse_Locations : tensorflow.Tensor of shape (N_communities, 2)
            A tensor containing the locations of the N_communities central points that are well distributed across the InputData_Locations.
        """
        # Define the number of nearest neighbors to consider
        k = tf.cast(tf.math.ceil(InputData_Locations.shape[0] / N_communities), tf.int32)

        # Calculate the Euclidean distances between all points in the data set
        distances = tf.reduce_sum(tf.square(tf.expand_dims(InputData_Locations, 1) - tf.expand_dims(InputData_Locations, 0)), axis=-1)

        # Find the indices of the nearest neighbors for each point
        _, nearest_neighbor_indices = tf.nn.top_k(-distances, k=k, sorted=True)
        
        # Gather the nearest neighbors for each point
        nearest_neighbor_Locations = tf.gather(InputData_Locations, nearest_neighbor_indices, axis=0)

        # Reshape the nearest neighbors tensor into the desired shape
        nearest_neighbor_Locations_reshaped = tf.reshape(nearest_neighbor_Locations, [-1, k, 2])

        # Pick every M-th point from the new data set
        # M = k because we want to pick one community from each set of nearest neighbors
        sparse_indices = tf.round(tf.linspace(0,tf.shape(nearest_neighbor_Locations_reshaped)[0]-1, N_communities))
        sparse_Locations = tf.cast(tf.gather(nearest_neighbor_Locations_reshaped, tf.cast(sparse_indices, tf.int32), axis=0), dtype=tf.float32)
        return tf.reduce_mean(sparse_Locations, axis=1)
        