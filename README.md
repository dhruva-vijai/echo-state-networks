# Echo State Networks

This repository contains the requisite files regarding a basic implementation of an Echo State Network, as described in the technical report, H. Jaeger(2001): The ”echo state” approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology, 2001.

## Problem Statement

Implement a simple echo state network, using the theoretical and practical frameworks provided in _H. Jaeger(2001): The ”echo state” approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology, 2001._ and _Lukoševičius, Mantas. “A Practical Guide to Applying Echo State Networks.” Neural Networks (2012)._ in Python from scratch and use the echo state network to implement a simple next-step time-series prediction method in order to analyze : 

1) Mathematical Functions as an approximation of non-stochastic time-series

2) The Mackey-Glass Dynamical System

## Background Information about Echo State Networks

Echo State Networks offer a simpler alternative to time-series prediction as compared to traditional recurrent networks by eliminating the need for computationally intensive backpropagation-through-time algorithms and succesive unrolling of the recurrent neural network at every time-step. In the classical ESN approach, the internal weights of the reservoir and input weights are randomly generated while the output linear readout weights are trained using any suitable linear regression algorithms in order to predict the output.

The internal reservoir is generated randomly, being randomly and sparsely connected in order to enable the development of complex dynamics in the reservoir. The input weights of the network are also initialized randomly, as are the linear readout weights. The internal state of the network is the vector of all activations of the internal reservoir and is initialized to zeroes.

__Units :__

1) U - Input Unit

2) X - Internal Unit

3) Y - Output Unit

__Weight Matrices :__

1) WIN - Input weight matrix of size NX * (1+NU)

2) W - Internal weight matrix of size NX * NX

3) WOUT - Output weught matrix of size NX * (1+NU+NY)

__Updation of The System :__

1) Output on a single Pass - y(n) = WOUT [1, x(n), y(n)]

2) Updated state - x(n+1) = (1-a).x(n) + a.tanh(WIN[1, u(n)] + W[x(n)])

a is the leaking/decay rate of the model during training and x(n),u(n) are the internal and input activations of the model

__Training The Model :__

The model is trained using any relevant linear regression algorithms with requisite noise injection during the process in order to stablize the model during training and prevent overfitting.


   
## Specifics of Implementation

## Necessary Libraries

1) NumPy

2) random

3) SciKitLearn

## Setup

cd projects

git clone https://github.com/dhruva-vijai/echo-state-networks/

python <implementation>.py

## Problems Faced



## Improvements



