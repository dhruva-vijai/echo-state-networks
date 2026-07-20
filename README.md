# Echo State Networks

This repository contains the requisite files regarding a basic implementation of an Echo State Network, as described in the technical report, H. Jaeger(2001): The ”echo state” approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology, 2001.

## Problem Statement

Implement a simple echo state network, using the theoretical and practical frameworks provided in _H. Jaeger(2001): The ”echo state” approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology, 2001._ and _Lukoševičius, Mantas. “A Practical Guide to Applying Echo State Networks.” Neural Networks (2012)._ in Python from scratch and use the echo state network to implement a simple next-step time-series prediction method in order to analyze : 

1) Mathematical Functions as an approximation of non-stochastic time-series

2) The Mackey-Glass Dynamical System (Not included yet due to problems with the code)

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

### Model Parameters for Periodic Sinusoidal Signal Testing

1) Number of Input Units = 2

2) Number of Internal Units = 1000

3) Number of Output Units = 2

4) Spectral Radius = 0.99

5) Scaling Factor = 20

6) Connectivity Fraction (Internal) = 0.0125

7) Decay Rate = 0.2

8) Transient Epochs = 300

9) Testing Sequence : sin(3x)+sin(x^2)


## Necessary Libraries

1) NumPy

2) random

3) SciKitLearn

## Setup

cd projects

git clone https://github.com/dhruva-vijai/echo-state-networks/

python "implementation".py

## Problems Faced

1) The addition of feedback loops for generative usage of the ESN has not yet been added and is not strictly necessary for predictive tasks, yet has had an influence on MSE degradation with epochs. Thus, feedback will also be introduced into the ESN in the next implementation.

2) Improper selection of model parameters (primarily spectral radius) adversely affects model performance


## Improvements

1) Implementing feedback loops for generative usage

2) Testing DeepESN frameworks or other, more modern reservoir computing frameworks

3) Tuning model hyperparameters (mainly spectral radius) for ideal model performance


## Images and Figures

### Testing the ESN on a Periodic Sinusoidal Signal For Next-Step Prediction

<img width="896" height="478" alt="Screenshot 2026-07-20 at 4 10 14 PM" src="https://github.com/user-attachments/assets/0678a76f-dfd2-4be3-989c-bc4db697b0e4" />


Above is a plot of MSE vs time-step. We see that MSE remains roughly consistent throughout the transient and teacher-forced periods but progressively degrades with time without any feedback addition.


<img width="883" height="479" alt="Screenshot 2026-07-20 at 4 10 45 PM" src="https://github.com/user-attachments/assets/1844ee95-aaa9-48f8-b61e-ea0d4f0b08ca" />


The above plot shows logarithmic MSE vs time-step in order to better analyze the model performance. It can be clearly seen that the model performance initially degrades during the transient period before settling down and improving through the teacher-forced period with MSE monotonically decreasing. After teacher-forcing is removed, it can be seen that the model MSE progressively degrades through future epochs without feedback addition.


<img width="868" height="461" alt="Screenshot 2026-07-20 at 4 11 04 PM" src="https://github.com/user-attachments/assets/7e585b4e-0807-4281-8a52-84fcc933df5c" />


The above graph shows the model and its' next step prediction on the given sinusoidal signal, with clearly demarcated transient, teacher-forced and free epochs.
