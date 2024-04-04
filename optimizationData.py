# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:50:29 2023

@author: Mels

This is a Python class that contains methods for optimization. The class has several attributes,
 including the number of iterations to perform during the optimization process, the costs array 
 to store the costs during training, the current iteration, and the weights for SES variance,
 positive population difference, population boundaries, and distance. It also has the regularization
 N powers, the TensorFlow optimizer to use in the optimization process, and methods for recording
 the cost values of the optimization process for plotting purposes and initializing population boundaries.
 The plotCosts method is used to plot the cost values over time.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class OptimizationData:
    def __init__(self, weights: list = [10,1,1,1], N_iterations: int = 100, LN: list[int] = [1,1,1,1]):
        '''
        Initializes an instance of the OptimizationData class.

        Parameters
        ----------
        weights : A list of the weights. Respectively, SESvariance, PopBounds, distance. Default is [10,1,1,1].
        N_iterations : Number of iterations to perform in the optimization process. Default is 100.
        LN : Regularization N powers. Default is [1,2,1,1].
            
        Attributes
        ----------
        N_iterations : int
            Number of iterations to perform in the optimization process.
        costs : ndarray of shape (N_iterations, 5)
            Array to store costs during training.
        i_iteration : int
            Current iteration.
        weight_SESvariance : int
            Weight for SES variance.
        weight_popBounds : int
            Weight for population boundaries.
        weight_distance : int
            Weight for distance.
        weight_education : int
            Weight for education.
        LN : list of int
            Regularization N powers.
        optimizer : TensorFlow optimizer
            TensorFlow optimizer to use in the optimization process.
        '''
        
        if len(LN)!=len(weights): raise ValueError("Input LN and weights should be equal in size")
        self.N_iterations = N_iterations
        self.costs = np.zeros((N_iterations, len(weights)+1)) # Define variables to store costs during training
        self.i_iteration = 0  # current interation
        
        # weights 
        self.weight_SESvariance = weights[0]
        self.weight_popBounds = weights[1]
        self.weight_distance = weights[2]
        self.weight_education = weights[3]
        
        # normalisation factors
        self.norm_SESvariance = 1
        self.norm_popBounds = 1
        self.norm_distance = 1
        self.norm_education = 1
        
        self.LN = LN # the reguralization N powers
        
        
    @tf.function
    def saveCosts(self, SES_variance: tf.Tensor, cost_popBounds: tf.Tensor, cost_distance: tf.Tensor, cost_education: tf.Tensor):
        '''
        Parameters:
            SES_variance : TensorFlow tensor
                Variance of the Socioeconomic Status of the communities.
            cost_popBounds : TensorFlow tensor
                Cost due to the number of individuals in each community.
            cost_distance : TensorFlow tensor
                Cost due to the distance between each community.
            cost_education : TensorFlow tensor
                Cost due to the education differences between each community.
        '''

        self.Cost_SES_variance = self.weight_SESvariance * abs( SES_variance / self.norm_SESvariance ) **self.LN[0]
        self.Cost_popBounds = self.weight_popBounds * abs( cost_popBounds / self.norm_popBounds ) **self.LN[1]
        self.Cost_distance = self.weight_distance * abs( cost_distance / self.norm_distance )**self.LN[2]
        self.cost_education = self.weight_education  * abs( cost_education / self.norm_education )**self.LN[3]
        
        
    @property
    def totalCost(self):
        return self.Cost_SES_variance + self.Cost_popBounds + self.Cost_distance + self.cost_education
        
    
    @tf.function
    def storeCosts(self):
        '''
        Stores the cost values of the optimization process for plotting purposes.
        
        Attributes:
            self.LN (list of ints):
                The regularization N powers.
            self.costs (numpy.ndarray):
                A (N_iterations x 5) numpy array to store the cost values during the training process.
            self.i_iteration (int):
                The current iteration of the optimization process.
            self.Cost_SES_variance (TensorFlow tensor):
                The cost due to SES variance.
            self.Cost_popBounds (TensorFlow tensor):
                The cost due to the number of individuals in each community.
            self.Cost_distance (TensorFlow tensor):
                The cost due to the distance between each community.
        '''
        if self.i_iteration < self.N_iterations:
            self.costs[self.i_iteration, 0] = self.totalCost
            self.costs[self.i_iteration, 1] = self.Cost_SES_variance.numpy()
            self.costs[self.i_iteration, 2] = self.Cost_popBounds.numpy()
            self.costs[self.i_iteration, 3] = self.Cost_distance.numpy()
            self.costs[self.i_iteration, 4] = self.cost_education.numpy()
        else:
            costs = np.array([self.totalCost, self.Cost_SES_variance.numpy(), 
                              self.Cost_popBounds.numpy(), self.Cost_distance.numpy(),
                              self.cost_education.numpy() ])
            self.costs = np.append(self.costs, costs[None,:], axis=0)
        self.i_iteration += 1
        
    
    @tf.function
    def initialize_popBoundaries(self, avg_pop, population_bounds: list=[0.8, 1.2]):
        self.population_bounds = tf.Variable(population_bounds, trainable=False, dtype=tf.float32) # Boundaries by which the population can grow or shrink of their original size
        self.popBoundHigh = self.population_bounds[1] * avg_pop # Upper population boundary
        self.popBoundLow = self.population_bounds[0] * avg_pop # Lower population boundary)
        
        
    def plotCosts(self):
        # Plot cost values over time
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(self.costs[:, 0], label="Total costs", ls="-")
        ax.plot(self.costs[:, 1], label="L"+str(self.LN[0])+" SES variance", ls="--")
        ax.plot(self.costs[:, 2], label="L"+str(self.LN[1])+" population bounds", ls="--")
        ax.plot(self.costs[:, 3], label="L"+str(self.LN[2])+" distance", ls="--")
        ax.plot(self.costs[:, 4], label="L"+str(self.LN[3])+" education", ls="--")
        ax.set_xlim(0, self.i_iteration-1)
        ax.set_ylim(0, np.max(self.costs[:, 0])*1.2)
        ax.set_title("Costs during Refinement")
        #ax.set_xlabel("Iterations")
        #ax.set_ylabel("Costs")
        plt.legend()
        return fig, ax
    
    def printCosts(self, text="Partial costs:"):
        print(text+"\n   L{} SES variance = {}\n   L{} population bounds = {},\n   L{} distance = {},\n   L{} education = {}\n".format(
            self.LN[0], self.Cost_SES_variance.numpy(),
            self.LN[1], self.Cost_popBounds.numpy(),
            self.LN[2], self.Cost_distance.numpy(),
            self.LN[3], self.cost_education.numpy()
        ))