# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:15 2023

@author: Mels
"""
import numpy as np
import tensorflow as tf
import pandas as pd

tf.config.run_functions_eagerly(True)  ### TODO should be False

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
        
    '''
    def __init__(self, Data, Population_size, num_communities):
        super(Dataset, self).__init__()
        self.Data = tf.Variable(Data[:,None], trainable=False, dtype=tf.float32)
        self.population_size = tf.Variable(Population_size[:,None], trainable=False, dtype=tf.float32)
        self.num_communities = tf.Variable(num_communities, trainable=False, dtype=tf.float32)
        #self.Map = tf.Variable(tf.eye(self.Data.shape[0]), trainable=True, dtype=tf.float32)
        #self.Map = tf.Variable(tf.ones((self.num_communities, self.Data.shape[0]), dtype=tf.float32), trainable=True)
        #TODO figure out how to make a non square matrix in entropy calculations

        # The matrix used to calculate entropy, and their original values
        self.Pk = tf.matmul(self.Data , tf.ones([self.Data.shape[1], self.Data.shape[0]]) )
        self.Pk.trainable = False
        
        # population parameters
        self.tot_pop = tf.reduce_sum(self.population_size)
        self.population_bounds = tf.Variable([.8, 1.2], trainable=False, dtype=tf.float32) # the boundaries by which the population can grow or shrink of their original size
        self.popBoundHigh  = self.population_bounds[1] * self.population_size
        self.popBoundLow  = self.population_bounds[0] * self.population_size
        
    
    @tf.function
    def call(self, inputs):
    # transforms the inputs according to the Map
        #TODO make it such that the SES values are multiplied with the population distribution, not just the map
        return tf.matmul(self.normalise_map(), inputs)
        #return tf.matmul(self.population_Map(), inputs)
    
    
    @tf.function
    def calculate_entropy(self, Plogits):
    # takes in the (mapped) matrix Pk and creates Qk 
    # then calculates the entropy via sum( Pk*log(Pk/Qk) )
        #TODO fix the entropy for non-positive logits
        Qlogits = tf.transpose(Plogits)
        return tf.reduce_sum( Plogits * tf.math.log(Plogits / Qlogits) )
    
    
    @tf.function
    def calculate_loss(self, Plogits):
    # Total loss with regularization
        # Original loss function (entropy)
        original_loss = self.calculate_entropy(Plogits)
        
        # L1 regularization term to let the map be positive
        L1_pos = tf.reduce_sum(tf.abs(tf.where(self.Map < 0., self.Map, 0.)))
        
        # L1 regularization term for population limits
        mapPop = self(self.population_size)    # the mapped population size
        popCost = tf.reduce_sum(self.Map, axis = 1) / self.Map.shape[1]  # the cost for erring
        L1_pop = tf.reduce_sum(tf.where(mapPop > self.popBoundHigh, popCost, 0)+
                               tf.where(mapPop < self.popBoundLow, popCost, 0))
        '''
        mapPop = tf.reduce_sum(self.population_Map(), axis=0)    # the mapped population size
        popCost = tf.reduce_sum(self.Map, axis = 1) / self.Map.shape[1]  # the cost for erring
        L1_pop = tf.reduce_sum(tf.where(mapPop > self.popBoundHigh, popCost, 0)+
                               tf.where(mapPop < self.popBoundLow, popCost, 0))
        '''
        # total L1  reegularization
        L1_reg = L1_pos + L1_pop
        
        #TODO add something about distances here
        
        return original_loss + L1_reg
    
    
    @tf.function
    def normalise_map(self):
        Map = self.Map / tf.reduce_sum(self.Map, axis=0)       # Divide each row by its sum
        return Map
    
    @tf.function
    def mapped_population_size(self):
        return self(self.population_size)
    
    @tf.function
    def mapped_Data(self):
        return self(self.Data)
    
    @tf.function
    def population_Map(self):
        return self.population_size[:,0] * self.normalise_map()
    

    @tf.function
    def train_step(self, input_data):
        with tf.GradientTape() as tape:
            logits = self(input_data)
            loss_value = self.calculate_loss(logits)
        grads = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value


#%% load data
# https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# should be downloaded as "CVS met statistische symbolen"
SES = pd.read_csv("Data/SES_WOA_scores_per_wijk_en_buurt_18012023_185321.csv"
                            , delimiter=';',quotechar='"').values
wijk = SES[:,1]
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)       # aantal particuliere huishoudens
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)                # sociaal economische waarde van regio
Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)              # numerieke maat van ongelijkheid in een regio

SES_WOA10 = tf.exp(SES_WOA)         # transform to Log scale to make all values positive
#TODO this will not really do anything? Because entropy uses log terms, will this not just reduce the calculation to being a simple difference?

N=4
model = Dataset(SES_WOA10[:N], part_huishoudens[:N], num_communities=3)

#%%
#age = tf.Variable(np.array([40,45,30,55]), dtype=tf.float32)
#pop_dense = tf.Variable(np.array([500,800,1200,400]), dtype=tf.float32)
#model = Dataset(age, pop_dense)
input_data = model.Pk
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
for i in range(500):
    loss_value = model.train_step(input_data)
    if i % 100 == 0:
        print("Step: {}, Loss: {}".format(i, loss_value.numpy()))
      
        
#%% output
print(
  "\nThe Map is:\n",tf.round( model.normalise_map() * 100 , 1 ).numpy(),
  "\n\nwhich counts up to:\n",tf.reduce_sum(tf.round( model.normalise_map() * 100 , 1 ).numpy(), axis=0),
  "\nThe Population Map is:\n",tf.round( model.population_Map()).numpy(),
  "\n\nSES_WOA:\n", model.mapped_Data().numpy(),
  "\n\nPopulation Size:\n", tf.round( model.mapped_population_size() ).numpy()
  )