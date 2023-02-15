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
    The class dataset is able to take a dataset defined below and do calculations with it.
    
    Variables: #TODO this contains a lot of text because I was thinking out loud. Thin this out
        Map: NxN float matrix.
            Maps the Data to the new definitions of neighbourhoods
            this means we will not map geographically yet, but by using percentages 
            of the total population in the neighbourhood. 
            For this reason, Map should be restricted to have all its rows count 
            up to 1.
        Data: Nx1xM float matrix.
            N is the amount of neighbourhoods defined in the dataset
            M is the amount of parameters we will use to optimise. So in the case 
            that we optimise the age, socio-economic value and diversity rating, 
            we have 3 parameters so M=3. 
        Pk, Qk: NxNxM float matrix.
            Pk contains the Nx1xM matrix Data N times. Qk is its transpose. 
            This means that Pk can be mapped in the same manner as Data.
            It also means that the matrices will look like
                 Pk = [A_1 A_2 A_3    Qk = [A_1 B_1 C_1
                       B_1 B_2 B_3          A_2 B_2 C_2
                       C_1 C_2 C_3]         A_3 B_3 C_3]
            and therefore they can be used to calculate the cross entropy.
            
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
            put in a certain group and optimise that. Entropy works on probabilities
        Important to note, lim_x->0 x*ln(x) = 0 (lHopital). However, one would still
            need to map the SES_WOA to fit in an interval of [0,1]
        I think it would be better to calculate the variance of the SES_WOA
        
        
    Creating the communities
        In this case we totally assume the the geographical locations of the neighbourhoods
            are the centerpoints. This is false.
        Right now we use those locations to extrapolate where the locations of the new
            communities should be in initialise_communities.
        This should ofcourse be better clarified. Center locations just do not represent
            what we are interested in.
            
    
    Calculating the distances
        We will work with a matrix that is representative of the distances between the
            neighbourhood_locs and the community_locs 
        
    '''
    def __init__(self, SES_WOA, Population_size, N_communities, neighbourhood_locs):
        super(Dataset, self).__init__()
        # initialise input
        self.SES_WOA = tf.Variable(SES_WOA[:,None], trainable=False, dtype=tf.float32)
        self.population_size = tf.Variable(Population_size[:,None], trainable=False, dtype=tf.float32)
        self.neighbourhood_locs = tf.Variable(neighbourhood_locs, trainable=False, dtype=tf.float32)
        
        # initialise parameters
        self.N_communities = N_communities
        self.N_neighbourhoods = self.SES_WOA.shape[0]
        self.Map = self.initialise_map()
        
        # create the center points on the new communities 
        if self.N_communities == self.N_neighbourhoods:
            self.community_locs = tf.Variable(neighbourhood_locs, trainable=False, dtype=tf.float32)
        elif self.N_communities < self.N_neighbourhoods:
            self.community_locs = self.initialise_communities(self.N_communities)
        else:
            #TODO This only works for N_communities <= N. Make it work for the other way
            raise Exception("Model is not able to create more communities than were originally present!")
        
        # initialise the distance matrix 
        self.initialise_distances()
        
        # The matrix used to calculate entropy, and their original values
        #self.Pk = tf.matmul(self.SES_WOA , tf.ones([self.SES_WOA.shape[1], self.N_communities]) )
        #self.Pk.trainable = False
        
        # population parameters
        self.population_bounds = tf.Variable([.8, 1.2], trainable=False, dtype=tf.float32) # the boundaries by which the population can grow or shrink of their original size
        self.tot_pop = tf.reduce_sum(self.population_size)
        self.avg_pop = self.tot_pop / self.N_communities
        self.popBoundHigh  = self.population_bounds[1] * self.avg_pop
        self.popBoundLow  = self.population_bounds[0] * self.avg_pop
        
    
    @tf.function
    def call(self, inputs):
        '''transforms the inputs according to the Map'''
        #TODO make it such that the SES values are multiplied with the population distribution, not just the map
        return tf.matmul(self.normalise_map(), inputs)
    
    
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
    def calculate_loss(self):
    # Total loss with regularization
        # Original loss function (entropy)
        SES_variance = tf.math.reduce_variance( self(self.SES_WOA) )
        
        # L1 regularization term to let the map be positive
        L2_popPositive = tf.reduce_sum(tf.abs(tf.where(self.Map < 0., self.Map, 0.)))
        
        # L1 regularization term for population limits
        mapPop = self(self.population_size)    # the mapped population size
        popCost = tf.reduce_sum(self.Map, axis = 1) / self.Map.shape[1]  # the cost for erring
        L1_popBounds = tf.reduce_sum(tf.where(mapPop > self.popBoundHigh, popCost, 0)+
                               tf.where(mapPop < self.popBoundLow, popCost, 0))
        
        # add regularisation from distances
        pop_distances = tf.multiply(self(self.population_size), self.distances)
        L1_dist = tf.reduce_sum(pop_distances / self.tot_pop / self.max_distance )
        
        # remember the partial costs
        self.SES_variance = SES_variance
        self.L2_popPositive = L2_popPositive
        self.L1_popBounds = L1_popBounds
        self.L1_dist = L1_dist
        
        return SES_variance + L2_popPositive + L1_popBounds #+ L1_dist
    
    
    @tf.function
    def initialise_map(self):
        """
        Initialise the Map matrix that maps the data. The map has the size
        (final number of communities x initial number of communities). When both are equal,
        the maps initial guess is an unitary matrix. In case this differs, the
        initial guess still tries to make it unitary, but either splits the remaining
        initial communities over the final communities, or the other way around.
    
        Returns:
            A TensorFlow variable with shape (N_communities, SES_WOA.shape[0]), 
            initialised with the desired values and set as trainable.
        """
        # initialise the Map matrix with zeros
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
            
        #TODO for some reason having the final amount of communities bigger makes calculations explode
        
        # Return the initialised Map matrix as a TensorFlow variable
        return tf.Variable(Map, dtype=tf.float32, trainable=True)
    
    
    @tf.function
    def normalise_map(self):
        '''
        normalises the map such that it does not create people or split people 
        over different communities
        '''
        Map = self.Map / tf.reduce_sum(self.Map, axis=0)       # Divide each row by its sum
        return Map
    
    @tf.function
    def mapped_population_size(self):
        return self(self.population_size)
    
    @tf.function
    def mapped_SES_WOA(self):
        return self(self.SES_WOA)
    
    @tf.function
    def population_Map(self):
        return self.population_size[:,0] * self.normalise_map()
    
    
    @tf.function
    def initialise_communities(self, N_communities):
        """
        Parameters
        ----------
        N_communities : int32
            The number of communities we want to end up with.

        Returns
        -------
        community_locs : float32 (N_communities x 2) array
            Array containing the grid locations of the newly created communities 
            
        This function uses KNN to initialise the locations of communities by sparsifying 
        the input locations.
        """
        # Define the number of nearest neighbors to consider
        k = tf.cast(tf.math.ceil(self.neighbourhood_locs.shape[0] / N_communities),tf.int32)

        # Calculate the Euclidean distances between all points in the data set
        distances = tf.reduce_sum(tf.square(tf.expand_dims(self.neighbourhood_locs, 1) - tf.expand_dims(self.neighbourhood_locs, 0)), axis=-1)

        # Find the indices of the nearest neighbors for each point
        _, nearest_neighbor_indices = tf.nn.top_k(-distances, k=k, sorted=True)

        # Gather the nearest neighbors for each point
        nearest_neighbors = tf.gather(self.neighbourhood_locs, nearest_neighbor_indices, axis=0)

        # Reshape the nearest neighbors tensor into the desired shape
        new_locs = tf.reshape(nearest_neighbors, [-1, k, 2])

        # Pick every M-th point from the new data set
        sparse_indices = tf.range(0, tf.shape(new_locs)[0], k)
        sparse_locs = tf.cast( tf.gather(new_locs, tf.cast(sparse_indices, tf.int32), axis=0), dtype=tf.float32)
        return tf.reduce_mean(sparse_locs, axis=1)
    
    
    @tf.function
    def initialise_distances(self):
        """
        Input
        neighbourhood_locs : float32 (N_communities x 2) array
            Array containing the grid locations of the initial neighbourhoods 
        community_locs : float32 (N_neighbourhoods x 2) array
            Array containing the grid locations of the newly created communities 

        Returns
        -------
        distances : float32 (N_neighbourhoods x N_communities) array
            Array containing the differences in distance between all indices
        """
        # Repeat the rows of neighbourhood_locs M times and the rows of community_locs N times
        neighbourhood_locs_repeated = tf.repeat(tf.expand_dims(self.neighbourhood_locs, axis=0), self.N_communities, axis=0)
        community_locs_repeated = tf.tile(tf.expand_dims(self.community_locs, axis=1), [1, self.N_neighbourhoods, 1])

        # Calculate the pairwise distances between all pairs of locations using the Euclidean distance formula
        self.distances = tf.sqrt(tf.reduce_sum(tf.square(neighbourhood_locs_repeated - community_locs_repeated), axis=-1))
        self.max_distance = tf.reduce_max(self.distances)
        

    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            loss_value = self.calculate_loss()
        grads = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value


#%% load data
# https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# should be downloaded as "CVS met statistische symbolen"
SES = pd.read_csv("Data/SES_WOA_scores_per_wijk_en_buurt_10022023_163026.csv"
                            , delimiter=';',quotechar='"').values
wijk = SES[:,1]
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)       # aantal particuliere huishoudens
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)                # sociaal economische waarde van regio
Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)              # numerieke maat van ongelijkheid in een regio
#TODO gebruik Spreiding in de berekening


# https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85231NED/table?ts=1669130108033
# should be downloaded as "CVS met statistische symbolen"
# select the correct areas and filter on 'vije tijd en cultuur - afstand tot bibleotheek'
# important to not is that you should only download neighbourhoods that maps.google recognises
# (https://nl.wikipedia.org/wiki/Buurten_en_wijken_in_Amsterdam so wijken but not buurten)
NabijheidCSV = pd.read_csv("Data/Nabijheid_voorzieningen__buurt_2021_10022023_162519.csv"
                            , delimiter=';',quotechar='"').values

# making sure it is ordered right
Nabijheid = np.zeros(wijk.shape)
for i in range(wijk.shape[0]):
    try:
        Nabijheid[i] = NabijheidCSV[np.where(NabijheidCSV==wijk[i])[0],1]
    except: raise Exception("NabijheidCSV is incomplete compared to the SES data.")
    
# locaties downloaden
city = "Amsterdam, Netherlands"
geolocator = Nominatim(user_agent="Dataset")
latlon0 = [ geolocator.geocode(city).latitude , geolocator.geocode(city).longitude ]
Locs = np.zeros([wijk.shape[0],2])
for i in range(wijk.shape[0]):
    loc = wijk[i]+", "+city
    geo_loc = geolocator.geocode(loc)
    
    if geo_loc==None:
        loc = wijk[i].replace("buurt","")+", "+city
        geo_loc = geolocator.geocode(loc)
        if geo_loc==None:
            loc = loc.replace("West","")
            loc = loc.replace("Zuid","")
            loc = loc.replace("Noord","")
            loc = loc.replace("Oost","")
            geo_loc = geolocator.geocode(loc)
        
    # putting all the coordinates in a grid
    latlon = [ geo_loc.latitude , geo_loc.longitude ]
    Locs[i,0] = distance.distance( latlon0 , [geo_loc.latitude, latlon0[1]] ).m
    Locs[i,0] = Locs[i,0] if (latlon0[0] - geo_loc.latitude > 0) else -Locs[i,0]
    Locs[i,1] = distance.distance( latlon0 , [latlon0[0], geo_loc.longitude] ).m
    Locs[i,1] = Locs[i,1] if (latlon0[1] - geo_loc.longitude > 0) else -Locs[i,1]


N=9
N_communities=5 
model = Dataset(SES_WOA[:N], part_huishoudens[:N], N_communities, Locs)

#%% plot 
img = plt.imread("Data/amsterdam.PNG")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-3000,4500, -2300, 2000])
ax.scatter(Locs[:,0],Locs[:,1])
ax.scatter(model.community_locs[:,0],model.community_locs[:,1])
ax.scatter(0,0)
plt.show()


#%% optimization
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
for i in range(800):
    loss_value = model.train_step()
    if i % 100 == 0:
        print("Step: {}, Loss: {}".format(i, loss_value.numpy()))
        print("partial costs: \n   SES_variance = ",model.SES_variance.numpy(),
              "\n   L2_popPositive = ",model.L2_popPositive.numpy(),"\n   L1_popBounds = ",
              model.L1_popBounds.numpy(),"\n   L1_dist = ",model.L1_dist.numpy())
      
        
#%% output
print(
  "\nThe Map is:\n",tf.round( model.normalise_map() * 100 , 1 ).numpy(),
  "\n\nwhich counts up to:\n",tf.reduce_sum(tf.round( model.normalise_map() * 100 , 1 ).numpy(), axis=0),
  "\nThe Population Map is:\n",tf.round( model.population_Map()).numpy(),
  "\n\nSES_WOA:\n", model.mapped_SES_WOA().numpy(),
  "\n\nPopulation Size:\n", tf.round( model.mapped_population_size() ).numpy()
  )

