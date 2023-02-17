# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:15 2023

@author: Mels
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy import distance

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
        self.Map = self.initialize_map() # Community map
        
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
        
        # Initialize the distance matrix
        self.initialize_distances()
        
        # The matrix used to calculate entropy, and their original values
        #self.Pk = tf.matmul(self.socioeconomic_data , tf.ones([self.socioeconomic_data.shape[1], self.N_communities]) )
        #self.Pk.trainable = False
        
        # Initialize population parameters
        self.population_bounds = tf.Variable([0.8, 1.2], trainable=False, dtype=tf.float32) # Boundaries by which the population can grow or shrink of their original size
        self.tot_pop = tf.reduce_sum(self.population_size)
        self.avg_pop = self.tot_pop / self.N_communities # Average population size
        self.popBoundHigh = self.population_bounds[1] * self.avg_pop # Upper population boundary
        self.popBoundLow = self.population_bounds[0] * self.avg_pop # Lower population boundary
        
        # Initialize the weights
        self.weight_SESvariance = 1
        self.weight_popPositive = 1
        self.weight_popBounds = 1
        self.weight_distance = 1 
        
        self.initialize_weights()
        
    
    @property
    def mapped_population_size(self):
        return self(self.population_size)
    
    @property
    def mapped_socioeconomic_data(self):
        return self(self.socioeconomic_data)
    
    @property
    def population_Map(self):
        return tf.round(self.population_size[:,0] * self.normalize_map())
    
    
    @tf.function
    def call(self, inputs):
        '''Transforms the inputs according to the map'''
        self.Map.assign(self.normalize_map()) # Normalize the community map
        return tf.matmul(self.Map, inputs)
    

    #@tf.function
    #def calculate_entropy(self, Plogits):
    #    '''
    #    takes in the (mapped) matrix Pk and creates Qk 
    #    then calculates the entropy via sum( Pk*log(Pk/Qk) )
    #    As lim_x->0 x*ln(x) = 0 (lHopital), we force this also.
    #    '''    
    #    Qlogits = tf.transpose(Plogits)
    #    Entropy = tf.reduce_sum( Plogits * tf.math.log(Plogits / Qlogits) )
    #    Entropy = tf.where(Plogits==0., Entropy, 0. )  # force to go to zero
    #    return Entropy
    
    
    @tf.function
    def cost_fn(self):
        # new populations
        mappedPopulation = self.mapped_population_size
        mappedPopulation_mat = self.population_Map
        
        # Calculate variance of socioeconomic data mapped to population map
        SES_variance = tf.math.reduce_variance(tf.matmul(self.normalize_map(), self.socioeconomic_data)*10) * self.weight_SESvariance
        #SES_variance = tf.math.reduce_variance(tf.matmul(mappedPopulation_mat, self.socioeconomic_data) / tf.reduce_sum(mappedPopulation/5)*10) * self.weight_SESvariance
        #SES_variance = tf.math.reduce_variance(tf.matmul(mappedPopulation_mat, self.socioeconomic_data) ) * self.weight_SESvariance
    
        # Regularization term to ensure population map is positive
        L2_popPositive = (tf.reduce_sum(tf.abs(tf.where(self.Map < 0., self.Map, 0.))) * self.weight_popPositive*100) ** 2
    
        # Regularization term for population limits
        L1_popBounds = tf.reduce_sum(tf.where(mappedPopulation > self.popBoundHigh, tf.abs(mappedPopulation-self.popBoundHigh), 0) +
                                     tf.where(mappedPopulation < self.popBoundLow, tf.abs(mappedPopulation-self.popBoundLow), 0)) * ( 
                                         self.weight_popBounds / self.tot_pop 
                                         )
    
        # Add regularization term based on distances
        pop_distances = tf.multiply(mappedPopulation_mat, self.distances)
        L1_distance = tf.reduce_sum(pop_distances / self.tot_pop / self.max_distance)*self.weight_distance
                
        # Record the partial costs for inspection
        self.SES_variance = SES_variance
        self.L2_popPositive = L2_popPositive
        self.L1_popBounds = L1_popBounds
        self.L1_distance = L1_distance
    
        # Return the sum of all partial costs
        return SES_variance + L1_popBounds + L2_popPositive + L1_distance
    
    
    @tf.function
    def train_step(self):
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
            tf.math.reduce_variance(tf.matmul(self.normalize_map(), self.socioeconomic_data)*10)
            #tf.math.reduce_variance(tf.matmul(self.population_Map, self.socioeconomic_data) )
            )
        self.weight_distance = self.weight_distance / ( 
            tf.reduce_sum(tf.multiply(self.population_Map, self.distances) / self.tot_pop / self.max_distance)
            )
    
    
    @tf.function
    def initialize_map(self):
        """
        Initialize the Map matrix that maps the data. The map has the size
        (final number of communities x initial number of communities). When both are equal,
        the maps initial guess is an unitary matrix. In case this differs, the
        initial guess still tries to make it unitary, but either splits the remaining
        initial communities over the final communities, or the other way around.
    
        Returns:
            A TensorFlow variable with shape (N_communities, N_neighbourhoods), 
            initialized with the desired values and set as trainable.
        """
        # TODO fill the map such that it spreads populations over nearest neighbours
        # Initialize the Map matrix with zeros
        Map = np.zeros([self.N_communities, self.N_neighbourhoods])
        
        # When the final number of communities is smaller than the initial number
        if self.N_communities < self.N_neighbourhoods:
            # Assign 1 to the diagonal elements
            for i in range(self.N_communities):
                Map[i, i] = 1.
            # Assign 1 / N_communities to the remaining elements
            Map[:, self.N_communities:] = 1 / self.N_communities
        else:
            # When the final number of communities is greater than the initial number
            # Calculate the factor by which we split the communities to spread them
            diff = self.N_communities - self.N_neighbourhoods
            factor = 1 / self.N_communities
            # Assign (1 - factor * diff) to the diagonal elements
            for i in range(self.N_neighbourhoods):
                Map[i, i] = 1. - factor * diff
            # Assign factor to the remaining elements
            Map[self.N_neighbourhoods:, :] = factor
            
        # Return the initialized Map matrix as a TensorFlow variable
        return tf.Variable(Map, dtype=tf.float32, trainable=True)
    
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
        sparse_indices = tf.range(0, tf.shape(nearest_neighbor_locs_reshaped)[0], k)
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
    

#%% load data
# Load SES data
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# Download the file named "CSV met statistische symbolen"
SES = pd.read_csv("Data/SES_WOA_scores_per_wijk_en_buurt_10022023_163026.csv", delimiter=';',quotechar='"').values

# Extract relevant columns
wijk = SES[:,1]  # Wijk (neighborhood) name
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)  # Number of private households
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)  # Socio-economic value of the region
Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)  # Numerical measure of inequality in a region
# TODO: use Spreiding in the calculation

# Load Nabijheid data
# Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85231NED/table?ts=1669130108033
# Download the file named "CSV met statistische symbolen"
# Select the correct areas and filter on 'vije tijd en cultuur - afstand tot bibleotheek'
# Note: only download neighborhoods that maps.google recognizes
# (see https://nl.wikipedia.org/wiki/Buurten_en_wijken_in_Amsterdam for neighborhoods that correspond to wijken)
NabijheidCSV = pd.read_csv("Data/Nabijheid_voorzieningen__buurt_2021_10022023_162519.csv", delimiter=';',quotechar='"').values

# Make sure Nabijheid data is in the correct order
Nabijheid = np.zeros(wijk.shape)
for i in range(wijk.shape[0]):
    try:
        Nabijheid[i] = NabijheidCSV[np.where(NabijheidCSV==wijk[i])[0],1]
    except:
        raise Exception("NabijheidCSV is incomplete compared to the SES data.")

# Download location data
city = "Amsterdam, Netherlands"
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode(city).latitude , geolocator.geocode(city).longitude ]
Locs = np.zeros([wijk.shape[0],2])
for i in range(wijk.shape[0]):
    loc = wijk[i]+", "+city
    geo_loc = geolocator.geocode(loc)
    
    # Handle cases where geolocation fails
    if geo_loc == None:
        # Try again without "buurt" in the name
        loc = wijk[i].replace("buurt","")+", "+city
        geo_loc = geolocator.geocode(loc)
        if geo_loc == None:
            # Try again without the directional prefixes ("West", "Zuid", "Noord", "Oost")
            loc = loc.replace("West","")
            loc = loc.replace("Zuid","")
            loc = loc.replace("Noord","")
            loc = loc.replace("Oost","")
            geo_loc = geolocator.geocode(loc)
        
    # Store latitude and longitude in a grid
    latlon = [ geo_loc.latitude , geo_loc.longitude ]
    Locs[i,0] = distance.distance( latlon0 , [geo_loc.latitude, latlon0[1]] ).m
    Locs[i,0] = Locs[i,0] if (latlon0[0] - geo_loc.latitude > 0) else -Locs[i,0]
    Locs[i,1] = distance.distance( latlon0 , [latlon0[0], geo_loc.longitude] ).m
    Locs[i,1] = Locs[i,1] if (latlon0[1] - geo_loc.longitude > 0) else -Locs[i,1]


#%% Define model parameters
N = 9 # Number of locations
N_communities = 5 # Number of communities
Niterations = 50 # Number of iterations for training

# Initialize model with preprocessed data
model = Dataset(SES_WOA[:N], part_huishoudens[:N], N_communities, Locs)

# Define optimization algorithm and learning rate
optimizer = tf.keras.optimizers.Adagrad(learning_rate=.1)

# Define variables to store costs during training
costs = np.zeros((Niterations, 5))


#%% Train the model for Niterations iterations
for i in range(Niterations):
    loss_value = model.train_step() # Take one optimization step
    if i % 10 == 0:
        # Print current loss and individual cost components
        print("Step: {}, Loss: {}".format(i, loss_value.numpy()))
        print("Partial costs:\n   SES variance = {},\n   L2 population positive = {},\n   L1 population bounds = {},\n   L1 distance = {}\n".format(
            model.SES_variance.numpy(),
            model.L2_popPositive.numpy(),
            model.L1_popBounds.numpy(),
            model.L1_distance.numpy()
        ))
    # Record cost values for plotting
    costs[i, 0] = loss_value
    costs[i, 1] = model.SES_variance.numpy()
    costs[i, 2] = model.L2_popPositive.numpy()
    costs[i, 3] = model.L1_popBounds.numpy()
    costs[i, 4] = model.L1_distance.numpy()


#%% Plot cost values over time
fig1, ax1 = plt.subplots()
ax1.plot(costs[:, 0]-costs[:, 2], label="Total costs", ls="-")
ax1.plot(costs[:, 1], label="SES variance", ls="--")
ax1.plot(costs[:, 3], label="L1 population bounds", ls="--")
ax1.plot(costs[:, 4], label="L1 distance", ls="--")
ax1.plot(costs[:, 2], label="L2 population positive", ls=":")
ax1.set_xlim(0, Niterations)
ax1.set_ylim(0, np.max(costs[:, 0]-costs[:, 2])*1.2)
plt.legend()


# histogram of the economic data
fig2, ax2 = plt.subplots()
num_bins = SES_WOA.shape[0]*2
bin_edges = np.linspace(np.min(np.append(SES_WOA,model.mapped_socioeconomic_data)), 
                        np.max(np.append(SES_WOA,model.mapped_socioeconomic_data)), num_bins+1)
n, bins, patches = ax2.hist(SES_WOA, bins=bin_edges, alpha=0.5, label='SES_WOA', density=True)
n, bins, patches = ax2.hist(model.mapped_socioeconomic_data.numpy().flatten(), 
                            bins=bin_edges, alpha=0.5, label='Mapped SES', density=True)
ax2.legend(loc='upper right')
ax2.set_xlabel('SES')
ax2.set_ylabel('Frequency')
ax2.set_title('Comparison of the economic data and its optimised form')
plt.show()


# Plot the population on a map of Amsterdam
img = plt.imread("Data/amsterdam.PNG")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-3000, 4500, -2300, 2000])
ax.scatter(Locs[:, 0], Locs[:, 1], s=part_huishoudens/100, alpha=1, label="Neighbourhoods") # Plot locations
ax.scatter(model.community_locs[:, 0], model.community_locs[:, 1], s=model.mapped_population_size/100, 
           alpha=1, label="Communities") # Plot community locations
ax.scatter(0, 0, alpha=.7) # Plot origin
plt.legend()
plt.show()

print(
  "\nThe Map is:\n",tf.round( model.normalize_map() * 100 , 1 ).numpy(),
  "\n\nwhich counts up to:\n",tf.reduce_sum(tf.round( model.normalize_map() * 100 , 1 ).numpy(), axis=0),
  "\n\nThe Population Map is:\n",tf.round( model.population_Map.numpy()),
  "\n\nsocioeconomic_data:\n", model.mapped_socioeconomic_data.numpy(),
  "\n\nPopulation Size:\n", tf.round( model.mapped_population_size ).numpy()
  )

