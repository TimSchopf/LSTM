# RNN19Assignmemt
Character-based language model with Long Short-Term Memory

This Repository contains the solution for the computer science exercise of the summer term 19 "Deep Learning & Neural Networks" Lecture @ Karlsruhe Institute of Technology from Alexander Waibel, Kay Rotmann and Ngoc-Quan Pham.  

## Task

Implementation of a character-level language model using Long Short-Term Memory Recurrent Neural Networks. An example implementation of a Vanilla RNN and an LSTM template from Ngoc-Quan Pham is given. Forward and backward passes as well as sampling have to be implemented in the LSTM.  

## Run

First, ensure that you are in the same directory with the python files and the "data" directory with the "input.txt" inside.

For the RNNs you can run two things:  

1. Train the RNN to see the loss function and the samples being generated every 1000 (elman) or 100 (LSTM) steps. You can manually change the hyperparameters to play around with the code a little bit.  

in terminal run:  
python elman-rnn.py train  
or  
python lstm.py train

2. Check the gradient correctness. This step is normally important when implementing back-propagation. The idea of grad-check is actually very simple.    
We need to know how to verify the correctness of the back-prop implementation.  
In order to do that we rely on comparison with the gradients computed using numerical differentiation.  
For each weight in the network we will have to do the forward pass twice (one by increasing the weight by \delta, and one by decreasing the weight by \delta).  
The difference between two forward passes gives us the gradient for that weight (maybe the code will be self-explanationable)

in terminal run:  
python elman-rnn.py gradcheck  
or  
python lstm.py gradcheck  
