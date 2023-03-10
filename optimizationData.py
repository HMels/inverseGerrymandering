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
    def __init__(self, weights=[10,1,1,1], N_iterations=100, LN=[1,2,1,1], optimizer=tf.keras.optimizers.Adamax(learning_rate=.1)):
        '''
        Initializes an instance of the OptimizationData class.

        Parameters
        ----------
        weights : list of int, optional
            A list of the weights. Respectively, SESvariance, popPositive, PopBounds, distance.
            The default is [10,1,1,1].
        N_iterations : int, optional
            The number of iterations to perform in the optimization process. The default is 100.
        LN : list of int, optional
            The regularization N powers. The default is [1,2,1,1].
        optimizer : TensorFlow optimizer, optional
            The TensorFlow optimizer to use in the optimization process. The default is tf.keras.optimizers.Adamax(learning_rate=.1).

        Attributes
        ----------
        N_iterations : int
            The number of iterations to perform in the optimization process.
        costs : ndarray of shape (N_iterations, 5)
            An array to store costs during training.
        i_iteration : int
            The current iteration.
        weight_SESvariance : int
            The weight for SES variance.
        weight_popPositive : int
            The weight for positive population difference.
        weight_popBounds : int
            The weight for population boundaries.
        weight_distance : int
            The weight for distance.
        LN : list of int
            The regularization N powers.
        optimizer : TensorFlow optimizer
            The TensorFlow optimizer to use in the optimization process.
        
        '''
        self.N_iterations = N_iterations
        self.costs = np.zeros((N_iterations, 5)) # Define variables to store costs during training
        self.i_iteration = 0  # current interation
        
        self.weight_SESvariance = weights[0]
        self.weight_popPositive = weights[1]
        self.weight_popBounds = weights[2]
        self.weight_distance = weights[3]
        
        self.LN = LN # the reguralization N powers
        self.optimizer = optimizer
        
        
    @tf.function
    def saveCosts(self, SES_variance, cost_popPositive, cost_popBounds, cost_distance):
        '''
        Parameters:
            SES_variance (TensorFlow tensor):
                The variance of the Socioeconomic Status of the communities.
            cost_popPositive (TensorFlow tensor):
                The cost due to the percentage of positive cases of COVID-19 in each community.
            cost_popBounds (TensorFlow tensor):
                The cost due to the number of individuals in each community.
            cost_distance (TensorFlow tensor):
                The cost due to the distance between each community.
        '''
        self.Cost_SES_variance = ( SES_variance * self.weight_SESvariance ) **self.LN[0]
        self.Cost_popPositive = ( cost_popPositive * self.weight_popPositive ) **self.LN[1]
        self.Cost_popBounds = ( cost_popBounds * self.weight_popBounds ) **self.LN[2]
        self.Cost_distance = ( cost_distance * self.weight_distance )**self.LN[3]
        
        
    @property
    def totalCost(self):
        return self.Cost_SES_variance + self.Cost_popPositive + self.Cost_popBounds + self.Cost_distance
        
    
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
            self.Cost_popPositive (TensorFlow tensor):
                The cost due to the percentage of positive cases of COVID-19 in each community.
            self.Cost_popBounds (TensorFlow tensor):
                The cost due to the number of individuals in each community.
            self.Cost_distance (TensorFlow tensor):
                The cost due to the distance between each community.
        '''
        if self.i_iteration < self.N_iterations:
            self.costs[self.i_iteration, 0] = self.totalCost
            self.costs[self.i_iteration, 1] = self.Cost_SES_variance.numpy()
            self.costs[self.i_iteration, 2] = self.Cost_popPositive.numpy()
            self.costs[self.i_iteration, 3] = self.Cost_popBounds.numpy()
            self.costs[self.i_iteration, 4] = self.Cost_distance.numpy()
        else:
            costs = np.array([self.totalCost,self.Cost_SES_variance.numpy(),
                self.Cost_popPositive.numpy(),self.Cost_popBounds.numpy(), self.Cost_distance.numpy() ])
            self.costs = np.append(self.costs, costs[None,:], axis=0)
        self.i_iteration += 1
        
    
    @tf.function
    def initialize_popBoundaries(self, avg_pop, population_bounds=[0.8, 1.2]):
        self.population_bounds = tf.Variable(population_bounds, trainable=False, dtype=tf.float32) # Boundaries by which the population can grow or shrink of their original size
        self.popBoundHigh = self.population_bounds[1] * avg_pop # Upper population boundary
        self.popBoundLow = self.population_bounds[0] * avg_pop # Lower population boundary)
        
        
    def plotCosts(self):
        # Plot cost values over time
        fig, ax = plt.subplots()
        ax.plot(self.costs[:, 0]-self.costs[:, 2], label="Total costs", ls="-")
        ax.plot(self.costs[:, 1], label="SES variance", ls="--")
        ax.plot(self.costs[:, 3], label="L1 population bounds", ls="--")
        ax.plot(self.costs[:, 4], label="L1 distance", ls="--")
        ax.plot(self.costs[:, 2], label="L2 population positive", ls=":")
        ax.set_xlim(0, self.i_iteration-1)
        ax.set_ylim(0, np.max(self.costs[:, 0]-self.costs[:, 2])*1.2)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Costs")
        plt.legend()
        return fig, ax
    
    def printCosts(self):
        print("Partial costs:\n   L{} SES variance = {},\n   L{} population positive = {},\n   L{} population bounds = {},\n   L{} distance = {}\n".format(
            self.LN[0], self.Cost_SES_variance.numpy(),
            self.LN[1], self.Cost_popPositive.numpy(),
            self.LN[2], self.Cost_popBounds.numpy(),
            self.LN[3], self.Cost_distance.numpy()
        ))