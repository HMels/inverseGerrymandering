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
        
    '''
    def __init__(self, Data):
        super(Dataset, self).__init__()
        self.Data = tf.Variable(Data, trainable=False, dtype=tf.float32)
        self.Map = tf.Variable(tf.eye(self.Data.shape[0]), trainable=True, dtype=tf.float32)
        
        # The matrix used to calculate entropy, and their original values
        self.Pk = tf.matmul(self.Data , tf.ones([Data.shape[1], Data.shape[0]]) )
        self.Pk.trainable = False
        
    
    @tf.function
    def call(self, inputs):
    # transforms the inputs according to the Map
        return tf.matmul(self.normalise_map(), inputs)
    
    
    @tf.function
    def calculate_entropy(self, Plogits):
    # takes in the (mapped) matrix Pk and creates Qk 
    # then calculates the entropy via sum( Pk*log(Pk/Qk) )
        Qlogits = tf.transpose(Plogits)
        return tf.reduce_sum( Plogits * tf.math.log(Plogits / Qlogits) )
    
    
    @tf.function
    def calculate_loss(self, Plogits):
    # Total loss with regularization
        # Original loss function (entropy)
        original_loss = self.calculate_entropy(Plogits)
        
        # L1 regularization term to let the map be positive
        L1_pos = tf.reduce_sum(tf.abs(tf.where(self.Map < 0., self.Map, 0.)))
        
        # L1 regularization term to let the map be normalised to one when summed up
        #L1_norm = tf.reduce_sum(tf.where(tf.reduce_sum(self.Map, axis=1, keepdims=True) > 1., self.Map, 0))
        
        # total L1  reegularization
        L1_reg = L1_pos #+ L1_norm
        
        return original_loss + L1_reg
    
    
    @tf.function
    def normalise_map(self):
        Map = self.Map / tf.reduce_sum(self.Map, axis=0)       # Divide each row by its sum
        return Map


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


Data = np.concatenate([part_huishoudens[:,None,None],SES_WOA[:,None,None]],axis=2) # use matrix projecting? to get in right format 


#%%
age = tf.Variable(np.array([40,45,30,55])[:,None], dtype=tf.float32)
model = Dataset(age)
input_data = model.Pk
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
for i in range(500):
    loss_value = model.train_step(input_data)
    if i % 100 == 0:
        print("Step: {}, Loss: {}".format(i, loss_value.numpy()))
        
print( tf.round( model.normalise_map() * 100 , 1 ) )