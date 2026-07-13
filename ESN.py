import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error 

random.seed(42)

class inputweights:
    def __init__(self,NU,NX,scaling,state=None): # NU - Number of Input units; NX - Number of reservoir units; Scaling is a factor implemented in order to control weighting of input
        self.NU = NU
        self.NX = NX
        self.scaling = scaling
        self.state=state

    def initialize(self):
        self.state = (np.random.rand(self.NX,1+self.NU)-0.5)*(self.scaling)  # The input weight matrix of size NX * 1 + NU is initialised randomly with a user-defined scale factor

    def returninputweights(self):  # Returns the weight matrix on function call
        return self.state

class internalweights:
    def __init__(self,NX,connectivity,spectral,state=None):  # NX - Number of Internal Units; Connectivity is the fraction of neuronal connections that are unweighted 
        self.NX = NX                                         # Sparse and random connectivity enables develpment of individual dynamics in the reservoir
        self.connectivity = connectivity                     # Spectral controls the desired spectral radius of the internal weight matrix and controls the degree of chaotic dynamics in the reservoir
        self.spectral = spectral
        self.state = state

    def initialize(self):                                    # The internal weight matrix is of size NX * NX
        array = np.random.rand(self.NX,self.NX)
        mask = np.random.rand(self.NX, self.NX) < self.connectivity
        array = np.random.rand(self.NX, self.NX) * mask
        spec = np.max(np.abs(np.linalg.eigvals(array)))
        array = (array*(self.spectral))/spec
        self.state = array

    def returninternalweights(self):
        return self.state

class outputweights:
    def __init__(self,NU,NX,NY,state=None):   # NU - Number of Input Units ; NX - Number of Internal Units ; NY - Number of output units
        self.NU = NU
        self.NX = NX
        self.NY = NY
        self.state = state

    def initialize(self):                                   # The output weight matrix is of size NY * ( 1 + NU +NX )
        self.state = np.zeros((self.NY,1+self.NU+self.NX))  # The output weights are initialized as all zeroes - Only these weights will be trained by using Ridge Regression

    def returnoutputweights(self):
        return self.state  
    
class internalstate:                       # The internal state is a vector of length NX which records the activations of the reservoir neurons
    def __init__(self,NX,state=None):
        self.NX = NX
        self.state = state

    def initialize(self): 
        self.state = np.zeros(self.NX)   # The internal state is initialized to zeroes at the first timestep and is then updated with every pass of the data through the model

    def returninternalstate(self):
        return self.state
    
class updation:
    def __init__(self,WIN,W,WOUT,input,internalstate,n,a): # Input - Current entered input ; n - Current Timestep ; a - Leaking/Decay Rate of Updation
        self.WIN = WIN
        self.W = W
        self.WOUT = WOUT
        self.input = input
        self.internalstate = internalstate
        self.a = a
        self.n = n

    def returnoutput(self):       
        conc = np.concatenate(([1],np.ravel(self.input),self.internalstate))   # Output on a single-pass through the network - Y = WOUT[1, u(n), x(n)] - concatenated
        new = (self.WOUT)@(conc)                                               # u(n) - Input activations ; x(n) - Internal activations
        return new
    
    def updatestate(self):
        term1 = (self.WIN)@(np.concatenate(([1],np.ravel(self.input))))
        term2 = (self.W)@(self.internalstate)
        update = np.tanh(term1+term2)
        self.internalstate = (1-self.a)*(self.internalstate)+(self.a)*(update)    # Updation of the internal state - x(n+1) = x(n).(1-a) + a.tanh(WIN[1, u(n)]+W[x(n)])

    def returnstate(self):
        return self.internalstate       # Returns the necessary updated internal state after a forward pass
    
class model:
    def __init__(self,NU,NX,NY,spectral,scaling,connectivity,a,state=None,WIN=None,W=None,WOUT=None):
        self.NU = NU
        self.NX = NX
        self.NY = NY
        self.spectral = spectral
        self.scaling = scaling
        self.connectivity = connectivity
        self.state = state
        self.a = a
        self.WIN = WIN
        self.W = W
        self.WOUT = WOUT

    def initialize(self):                                             # Initialize is a method that sets up the model and its' parameters 
        IW = inputweights(self.NU,self.NX,self.scaling)               # Parameters - Input Weights, Internal Weights, Output Weights, Internal State
        IW.initialize()
        self.WIN =IW.returninputweights()
        IntW = internalweights(self.NX,self.connectivity,self.spectral)
        IntW.initialize()
        self.W =IntW.returninternalweights()
        OW = outputweights(self.NU,self.NX,self.NY)
        OW.initialize()
        self.WOUT =OW.returnoutputweights()
        IntS = internalstate(self.NX)
        IntS.initialize()
        self.state =(IntS).returninternalstate()

    def trainpass(self,timestep,input):           # Trainpass describes the entire process by which one data point at a marked timestep is run through the entire network during training
        updateobject = updation(self.WIN,self.W,self.WOUT,input,self.state,timestep,self.a)
        updateobject.updatestate()
        self.state=updateobject.returnstate()
        return np.concatenate(([1],np.atleast_1d(input),self.state))      # Trainpass updates the internal state

    def trainrun(self,total,transient,input_sequence):  # Trainrun is a method which shows the entire training process including a transient period 
        d = []
        for i in range(total):
            v = self.trainpass(i,input_sequence[i])
            if i>= transient:
                d.append(v)
        return np.vstack(d)      # Returns a NumPy array containing the relevant input and output at every timestep

    def fit(self,total,transient,input_sequence,teacher_sequence):  # fit implements Ridge Regression using scikit-learn in order to train the linear readout
        x = self.trainrun(total,transient,input_sequence)
        x += np.random.normal(0, 1e-9, x.shape) 
        y = teacher_sequence[transient:total]
        readout = Ridge(alpha=1e-3, fit_intercept=False)      # Ridge regression penalizes the model for learning very large weights and prevents overfitting in the model
        readout.fit(x, y)
        self.WOUT = readout.coef_

# The testing methods used here are currently being used in next-step prediction - At every timestep, the model is fed the previous output in order to evaluate model accuracy

    def testpass(self,timestep,input):
        updateobject = updation(self.WIN,self.W,self.WOUT,input,self.state,timestep,self.a)     # Testpass describes the process by which the model passes data during testing
        updateobject.updatestate()
        self.state=updateobject.returnstate()
        d = updateobject.returnoutput()         # The testpass method returns the model output in order to feed it as the next input to enable next-step prediction
        return d
    
    def test(self,total,transient,initial):    # The test method shows the entire testing method without inclusiong any teacher-forcing and only includes a transient period
        d=[]
        input=initial                          # Teacher forcing refers to feeding the true input to the model for a duration in order to  'set it right'
        for i in range(total):
            v=self.testpass(i,input)
            if i>=transient:
                d.append(v)
            input=np.ravel(v)
        return np.vstack(d)
    
    def forced(self,total,transient,input_sequence):       # The forced method only uses a teacher-forcing-based approach by feeding the true input into the model
        d=[]
        for i in range(total):
            v=self.testpass(i,input_sequence[i])         # This method only forces the model and is not true testing
            if i>=transient:
                d.append(v)
        return np.vstack(d)
    
    def hybrid(self,total,transient,force,input_sequence):     # The hybrid method implements teacher forcing for a duration - force and removes a duration - transient
        d=[]
        for i in range(total):
            if i<force:
                v=self.testpass(i,input_sequence[i])
                input=v
            else:
                 v=self.testpass(i,input)
                 input=np.ravel(v)
            if i>=transient:
                d.append(v)
        return np.vstack(d)                                 # This method is the training method used for model testing/graph-based prediction
    
    # def evaluation(self,total,transient,force,input_sequence,teacher_sequence):  # The evaluation method is just an updated hybrid testing method that calculates MSE
    #     d=[]
    #     for i in range(total):
    #         if i<force:
    #             v=self.testpass(i,input_sequence[i])
    #             input=v
    #         else:
    #              v=self.testpass(i,input)
    #              input=np.ravel(v)
    #         if i>=transient:
    #             d.append(v)
    #     tempmse = mean_squared_error(teacher_sequence[:i],np.vstack(d))
    #     mse = mean_squared_error(teacher_sequence[transient:total],np.vstack(d))     # MSE - Mean Squared Error - enables evaluation of model accuracy
    #     return [np.vstack(d),mse,tempmse]         

    def evaluation(self, total, transient, force, input_sequence, teacher_sequence):
        d = []
        mselist = []

        for i in range(total):
            if i < force:
                v = self.testpass(i, input_sequence[i])
                input = v
            else:
                v = self.testpass(i, input)
                input = np.ravel(v)

            if i >= transient:
                d.append(v)

                pred = np.vstack(d)
                true = teacher_sequence[transient:transient + len(pred)]

                tempmse = mean_squared_error(true, pred)
                mselist.append(tempmse)

        pred = np.vstack(d)
        true = teacher_sequence[transient:transient + len(pred)]
        mse = mean_squared_error(true, pred)

        return pred, mse, mselist                                           
    

class tuner:             # The tuner class enables a randomized-search-based optimization of the hyperparameters of the ESN - The optimizing function here is the MSE
    def __init__(self,NU,NX,NY,total,transient,force,input_sequence,teacher_sequence,rangespectral,rangescale,rangeleak,runs):
        self.NU = NU
        self.NX = NX
        self.NY = NY
        self.total = total
        self.transient = transient
        self.force = force
        self.input_sequence = input_sequence
        self.teacher_sequence = teacher_sequence
        self.rangespectral = rangespectral
        self.rangescale = rangescale
        self.rangeleak = rangeleak
        self.runs = runs

    def randomrun(self):   # Defines a complete run with random hyperparameters
        spectral = random.uniform(*self.rangespectral)  # Range of values for model spectral radius from which a random choice is made
        scaling = random.uniform(*self.rangescale)     # Range of values for input scaling from which a random choice is made
        leak = random.uniform(*self.rangeleak)       # Range of values for leaking/decay rate from which a random choice is made
        instance = model(self.NU,self.NX,self.NY,spectral,scaling,0.0125,leak)
        instance.initialize()
        instance.fit(self.total-self.force, self.transient, self.input_sequence, self.teacher_sequence)
        comparison_teacher = (self.teacher_sequence)
        mse = (instance.evaluation(self.total, self.transient,self.force,self.input_sequence,comparison_teacher))[1]
        return [mse,[spectral,scaling,leak]]   # Returns the average MSE for a random run as well as the variable hyperparameters of the specific instance
    
    def tune(self):       # The tune method describes a complete tuning run 
        best = [10000000,[]]
        for i in range(self.runs):
            mse = self.randomrun()
            if mse[0]<best[0]:
                best = mse
        print(best)




    

