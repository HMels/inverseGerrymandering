# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:12:20 2023

@author: Mels

This is a Python script that defines a class called InputData. It has methods to 
reload data from a new CSV file and to map the geospatial data to a regular grid. 
The class also includes several attributes, such as neighbourhoods, Socioeconomic_data,
 Population, Locations, N, center_coordinates, and gdf, which store the loaded data in 
 various formats such as NumPy arrays, TensorFlow Tensors, and geopandas dataframes.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from geopy import distance
import tensorflow as tf


class InputData:
    def __init__(self, path):
        """
        Initializes an instance of the class and loads socio-economic data from a CSV file.
    
        Args:
            path (str): The path to the CSV file containing the socio-economic data.
        
        Raises:
            FileNotFoundError: If the specified file path does not exist.
    
        Notes:
            The CSV file must have columns separated by semicolons and enclosed in quotes.
            Rows with missing or invalid data are removed from the loaded data.
        
        Attributes:
            neighbourhoods (array):
                1D array of strings containing the names of the neighbourhoods.
            Population (Tensor):
                1D TensorFlow Tensor of floats containing the number of private Population in each neighbourhood.
            Socioeconomic_data (Tensor):
                1D TensorFlow Tensor of floats containing the socio-economic value of each neighbourhood.
        """
        
        # Load data from CSV file using pandas
        data = pd.read_csv(path, delimiter=';', quotechar='"', na_values='       .')
        
        # Replace '       .' with NaN in columns 3 and 4 and delete these rows as they don't have data data
        data = data.replace('       .', float('nan'))
        data.dropna(inplace=True)
        data = data.values

        # Extract relevant variables and store as class variables
        self.neighbourhoods = data[:,1]  # neighbourhood names
        self.Socioeconomic_data = tf.Variable(np.array(data[:,3].tolist()).astype(np.float32)[:, None],
                                              trainable=False, dtype=tf.float32) # Socio-economic value of the region
        self.Population = tf.Variable(np.array(data[:,2].tolist()).astype(np.float32)[:, None],
                                           trainable=False, dtype=tf.float32) # Number of households per neighbourhood
        self.Locations = None      # locations yet to be 
        self.N = self.Socioeconomic_data.shape[0]
        
        
    @property
    def Socioeconomic_population(self):
        '''
        Returns
        -------
        tf.float32
            The socioeconomic data multiplied by the population to get the actual socioeconomic value.

        '''
        return self.Socioeconomic_data * self.Population
        
        
    def reload_path(self, path):
        # Load data from CSV file using pandas
        data = pd.read_csv(path, delimiter=';', quotechar='"', na_values='       .')
        
        # Replace '       .' with NaN in columns 3 and 4 and delete these rows as they don't have data data
        data = data.replace('       .', float('nan'))
        data.dropna(inplace=True)
        data = data.values

        # Extract relevant variables and store as class variables
        self.neighbourhoods = data[:,1]  # neighbourhood names
        self.Socioeconomic_data = tf.Variable(np.array(data[:,3].tolist()).astype(np.float32)[:, None],
                                              trainable=False, dtype=tf.float32) # Socio-economic value of the region
        self.Population = tf.Variable(np.array(data[:,2].tolist()).astype(np.float32)[:, None],
                                           trainable=False, dtype=tf.float32) # Number of households per neighbourhood
        self.Locations = None      # locations yet to be 
        self.N = self.Socioeconomic_data.shape[0]
        
    
    def load_geo_data(self, filename):
        """
        Loads geospatial data from a specified file in geopackage format and stores it in the class variables. 
    
        Args:
            filename (str): The path to the geopackage file containing the geospatial data.
    
        Returns:
            None
    
        Raises:
            FileNotFoundError: If the specified file path does not exist.
    
        Notes:
            This function uses the geopandas library to read a geopackage file specified by `filename`. It then 
            converts the spatial reference system to 'EPSG:4326', explodes the multipolygons into polygons and 
            converts the polygons to center coordinates. The center coordinates are stored in the `center_coordinates`
            class variable as a numpy array, while the original geopandas dataframe is stored in the `gdf` class variable.
            The CSV file must have columns separated by semicolons and enclosed in quotes.
    
        Attributes:
            center_coordinates (numpy.ndarray): A 2D numpy array of shape (n, 2), where n is the number of polygons
            in the geopackage file. The first column represents the longitude of the center of each polygon, while the
            second column represents its latitude.
    
            gdf (geopandas.geodataframe.GeoDataFrame): A geopandas dataframe containing the geospatial data 
            loaded from the geopackage file.
        """
        
        # Load data from geopackage file using geopandas
        gdf = gpd.read_file(filename)
        gdf = gdf.to_crs('EPSG:4326') # Convert to longlat
        gdf = gdf.explode(index_parts=True)  # Convert multipolygon to polygon
        coords_list = [list(x.exterior.coords) for x in gdf.geometry] # Make a list out of it

        # Convert polygons to center coordinates and store as class variable
        center_coordinates = []
        for coords in coords_list:
            center_coordinates.append(np.average(coords, axis=0))
        center_coordinates = np.array(center_coordinates)
        center_coordinates[:, [0, 1]] = center_coordinates[:, [1, 0]]
        
        self.center_coordinates = tf.Variable(center_coordinates, trainable=False, dtype=tf.float32)
        self.gdf = gdf


    def map2grid(self, latlon0):
        """
        Maps the coordinates in `neighbourhood_coords` to a grid that follows the reference coordinate `latlon0`.
        
        Args:
            latlon0 (tuple): A tuple containing the reference coordinate in the form (latitude, longitude).
        
        Returns:
            Locations (Tensor): TensorFlow Tensor of shape (n,2) containing the mapped coordinates.
        
        Notes:
            This function calculates the distance in meters of each coordinate in `neighbourhood_coords` to the reference
            coordinate `latlon0`. It then maps the coordinates to a grid based on their relative position to the reference
            coordinate, with the reference coordinate as the origin of the grid. The resulting `Locations` array is returned.
        """        
        # Loop through all coordinates and store their latitude and longitude in a grid
        Locations = np.zeros(self.neighbourhood_coords.shape, dtype=np.float32)
        for i in range(self.neighbourhood_coords.shape[0]):
            loc = np.zeros(2, dtype=np.float32)
            loc[0] = distance.distance(latlon0, [self.neighbourhood_coords[i, 0], latlon0[1]]).m
            loc[0] = loc[0] if (latlon0[0] < self.neighbourhood_coords[i, 0]) else -loc[0]
            loc[1] = distance.distance(latlon0, [latlon0[0], self.neighbourhood_coords[i, 1]]).m
            loc[1] = loc[1] if (latlon0[1] < self.neighbourhood_coords[i, 1]) else -loc[1]
            Locations[i, :] = loc

        self.Locations = tf.Variable(Locations, trainable=False, dtype=tf.float32)
    

    def buurt_filter(self):
        """
        Filters the socio-economic data based on whether the neighbourhoods are present in the geopandas data and 
        stores the relevant data in class variables.
        
        Args:
            None
        
        Returns:
            None
            
        Notes:
            This function compares each neighbourhood in the socio-economic data to the `buurtnaam` column of the 
            geopandas dataframe stored in the `gdf` class variable. If a neighbourhood is present in the geopandas data,
            the index of the corresponding row is appended to the `index` list. The socio-economic value, number of
            Population, and name of the neighbourhood are also stored in separate lists. The lists are then converted to
            numpy arrays, and the `center_coordinates` class variable is filtered based on the `index` list. A warning
            message is printed for any neighbourhood in the socio-economic data that is not present in the geopandas data.
        
        Attributes:
            Socioeconomic_data (tf.Variable): A 1D numpy array of floats containing the socio-economic value of each 
            neighbourhood that is present in the geopandas data.
            
            Population (tf.Variable): A 1D numpy array of floats containing the number of private Population in each 
            neighbourhood that is present in the geopandas data.
            
            neighbourhoods (numpy.ndarray): A 1D numpy array of strings containing the names of the neighbourhoods that 
            are present in the geopandas data.
            
            neighbourhood_coords (tf.Variable): A 2D numpy array of shape (n, 2), where n is the number of neighbourhoods 
            present in the geopandas data. The first column represents the longitude of the center of each polygon, while 
            the second column represents its latitude.
        """
        
        index=[]
        Socioeconomic_data=[]
        Population=[]
        neighbourhoods=[]
        i=0
        print("WARNING: GDF CAN ONLY FIND BUURTEN AND NOT WIJKEN OR CITIES. THEREFORE, A LOT OF DATA WILL BE MISSING:")
        for loc in self.neighbourhoods:
            truefalse=False # used to check if the loc is found in the gdf file
            j = 0
            for buurt in self.gdf.buurtnaam:
                if buurt==loc:
                    truefalse=True
                    index.append(j)
                    Socioeconomic_data.append(self.Socioeconomic_data[i])
                    Population.append(self.Population[i])
                    neighbourhoods.append(self.neighbourhoods[i])
                    break
                else:
                    j+=1

            if not truefalse: # check if wijk was not found
                print("Warning:",loc,"has not been found in gdf data. Check if instance should have been found")

            i+=1

        # save all in arrays
        self.Socioeconomic_data = tf.Variable(Socioeconomic_data, trainable=False, dtype=tf.float32)
        self.Population = tf.Variable(Population, trainable=False, dtype=tf.float32)
        self.neighbourhoods = np.array(neighbourhoods)
        self.neighbourhood_coords = tf.gather(self.center_coordinates, index)
        self.N = self.Socioeconomic_data.shape[0]
