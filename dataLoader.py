# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:12:20 2023

@author: Mels
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from geopy import distance


class DataLoader:
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
            neighborhoods (numpy.ndarray): A 1D array of strings containing the names of the neighborhoods.
            households (numpy.ndarray): A 1D array of floats containing the number of private households in each neighborhood.
            socioeconomic_data (numpy.ndarray): A 1D array of floats containing the socio-economic value of each neighborhood.
        """
        
        # Load data from CSV file using pandas
        SES = pd.read_csv(path, delimiter=';', quotechar='"', na_values='       .')
        
        # Replace '       .' with NaN in columns 3 and 4 and delete these rows as they don't have SES data
        SES = SES.replace('       .', float('nan'))
        SES.dropna(inplace=True)
        SES = SES.values

        # Extract relevant variables and store as class variables
        self.neighborhoods = SES[:,1]  # Neighborhood names
        self.households = np.array(SES[:,2].tolist()).astype(np.float32)  # Number of private households
        self.socioeconomic_data = np.array(SES[:,3].tolist()).astype(np.float32)  # Socio-economic value of the region
    
    
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
        
        self.center_coordinates = center_coordinates
        self.gdf = gdf


    def map2grid(self, latlon0):
        """
        Maps the coordinates in coords to a grid that follows the reference coordinate latlon0.
        
        Parameters:
            Locs (numpy.ndarray): A numpy array of shape (n,2) containing the coordinates to map.
            latlon0 (tuple): A tuple containing the reference coordinate.
        
        Returns:
            numpy.ndarray: A numpy array of shape (n,2) containing the mapped coordinates.
        """
        
        # Loop through all coordinates and store their latitude and longitude in a grid
        neighbourhood_locs = np.zeros(self.neighbourhoods_coords.shape, dtype=np.float32)
        for i in range(self.neighbourhoods_coords.shape[0]):
            loc = np.zeros(2, dtype=np.float32)
            loc[0] = distance.distance(latlon0, [self.neighbourhoods_coords[i, 0], latlon0[1]]).m
            loc[0] = loc[0] if (latlon0[0] < self.neighbourhoods_coords[i, 0]) else -loc[0]
            loc[1] = distance.distance(latlon0, [latlon0[0], self.neighbourhoods_coords[i, 1]]).m
            loc[1] = loc[1] if (latlon0[1] < self.neighbourhoods_coords[i, 1]) else -loc[1]
            neighbourhood_locs[i, :] = loc

        self.neighbourhood_locs = neighbourhood_locs
    

    def buurt_filter(self):
        """
        Filters the socio-economic data based on whether the neighborhoods are present in the geopandas data and 
        stores the relevant data in class variables.
        
        Args:
            None
        
        Returns:
            None
            
        Notes:
            This function compares each neighborhood in the socio-economic data to the `buurtnaam` column of the 
            geopandas dataframe stored in the `gdf` class variable. If a neighborhood is present in the geopandas data,
            the index of the corresponding row is appended to the `index` list. The socio-economic value, number of
            households, and name of the neighborhood are also stored in separate lists. The lists are then converted to
            numpy arrays, and the `center_coordinates` class variable is filtered based on the `index` list. A warning
            message is printed for any neighborhood in the socio-economic data that is not present in the geopandas data.
        
        Attributes:
            socioeconomic_data (numpy.ndarray): A 1D numpy array of floats containing the socio-economic value of each 
            neighborhood that is present in the geopandas data.
            
            households (numpy.ndarray): A 1D numpy array of floats containing the number of private households in each 
            neighborhood that is present in the geopandas data.
            
            neighborhoods (numpy.ndarray): A 1D numpy array of strings containing the names of the neighborhoods that 
            are present in the geopandas data.
            
            coords (numpy.ndarray): A 2D numpy array of shape (n, 2), where n is the number of neighborhoods present 
            in the geopandas data. The first column represents the longitude of the center of each polygon, while the 
            second column represents its latitude.
        """
        
        index=[]
        socioeconomic_data=[]
        households=[]
        neighborhoods=[]
        i=0
        print("WARNING: GDF CAN ONLY FIND BUURTEN AND NOT WIJKEN OR CITIES. THEREFORE, A LOT OF DATA WILL BE MISSING:")
        for loc in self.neighborhoods:
            truefalse=False # used to check if the loc is found in the gdf file
            j = 0
            for buurt in self.gdf.buurtnaam:
                if buurt==loc:
                    truefalse=True
                    index.append(j)
                    socioeconomic_data.append(self.socioeconomic_data[i])
                    households.append(self.households[i])
                    neighborhoods.append(self.neighborhoods[i])
                    break
                else:
                    j+=1

            if not truefalse: # check if wijk was not found
                print("Warning:",loc,"has not been found in gdf data. Check if instance should have been found")

            i+=1

        # save all in arrays
        self.socioeconomic_data = np.array(socioeconomic_data)
        self.households = np.array(households)
        self.neighborhoods = np.array(neighborhoods)
        self.neighbourhoods_coords = self.center_coordinates[index,:]