# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:15:56 2023

@author: Mels
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, shgo, differential_evolution
from scipy.stats import entropy
import tensorflow as tf

tf.config.run_functions_eagerly(True)  ### TODO should be False

class dataset(tf.keras.Model):
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
    def __init__(self, Data, Map=None):
        super(dataset, self).__init__()
        self.Data = tf.Variable(Data, trainable=False, dtype=tf.float32)
        self.Map = ( tf.Variable(Map, trainable=True, dtype=tf.float32) if Map is not None 
            else tf.Variable(tf.eye(self.Data.shape[0]), trainable=True, dtype=tf.float32) )
        
        # The matrices used to calculate entropy, and their original values
        self.Pk = tf.matmul( tf.ones([Data.shape[0], Data.shape[1]]) , self.Data )
        #self.Qk = tf.transpose(self.Pk, (1, 0, 2))
        #self.Pk0 = self.Pk
        self.Pk.trainable = False
        #self.Pk0.trainable = False
        #self.Qk.trainable = False

    
    @tf.function
    def call(self):
        return tf.matmul(self.Map, self.Pk)
        
    
    @tf.function
    def map_data(self):
        return tf.transpose(tf.matmul(self.Map, tf.transpose(self.Data, (2,0,1)) ) , (1,0,2) )


    @tf.function(autograph=True)
    def calculate_entropy(self, Map):
        """
        Calculates the entropy of Data via sum( Pk * log( Pk / Qk ) )
        
        Output:
            entr: double 
                A single value that represents the total entropy in the mapped neighbourhoods.
        """
        Pk = tf.Variable( tf.matmul(Map, self.Pk) , trainable=False, dtype=tf.float32)
        Qk = tf.transpose(self.Pk, (1, 0, 2))
        #x = tf.Variable(tf.reduce_sum(Pk * tf.math.log(Pk / Qk)))
        #x = tf.reduce_sum(tf.square(Pk-Qk))
        return tf.Variable( tf.reduce_sum( Pk * tf.math.log(Pk / Qk) , axis=2))
    
    
def train_step(model):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_value = model.calculate_entropy(model.trainable_variables)
    grads = tape.gradient(loss_value, model.trainable_variables)
    print('grads=', grads)
    optimizer = tf.keras.optimizers.Adam()
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value


'''
#@tf.function 
def train(self, learning_rate=1, epochs=100, opt_fn=tf.optimizers.Adagrad, opt=None):
    """
    Training the optimizes on the data to find the right map.
    """
    if opt is None: opt=opt_fn(learning_rate)
    for i in range(epochs):
        # calculate and apply gradients
        loss = self.calculate_entropy()
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        if i%100==0 and i!=0: print('iteration='+str(i)+'/'+str(epochs))
    return loss
'''

#%% load data
# https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
# should be downloaded as "CVS met statistische symbolen"
SES = pd.read_csv("../SES_WOA_scores_per_wijk_en_buurt_18012023_185321.csv"
                            , delimiter=';',quotechar='"').values
wijk = SES[:,1]
part_huishoudens = np.array(SES[:,2].tolist()).astype(np.float32)       # aantal particuliere huishoudens
SES_WOA = np.array(SES[:,3].tolist()).astype(np.float32)                # sociaal economische waarde van regio
Spreiding = np.array(SES[:,4].tolist()).astype(np.float32)              # numerieke maat van ongelijkheid in een regio


Data = np.concatenate([part_huishoudens[:,None,None],SES_WOA[:,None,None]],axis=2) # use matrix projecting? to get in right format 


#%% load parameters
age = np.array([40,45,30,55])[:,None,None]
Data = dataset(age)

train_step(Data)


"lets try to make our own gradient"

Data1=dataset(age,Map=np.array([[1., 0., 0., 0.03],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., .97]]))
#Data1.calculate_entropy()
profit = tf.reduce_sum( Data.calculate_entropy()-Data1.calculate_entropy() )
print("this should be bigger than 0: ", profit.numpy())

"I could also make a super wrong example and then see if it can solve that, because there is a probability calculate entropy wrong and it's already optimal"