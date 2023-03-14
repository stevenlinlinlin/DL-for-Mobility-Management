# Deep Learning based Intelligent Connectivity for Mobility Management
Predict the user's next location with pytorch machine learning models

## Purpose
- explore how to use different deep learning techniques
- predict the user's next location
- prepare the pre-work for Handover in advance
- solve the degradation of service quality and energy loss caused by passive handover of traditional base stations
- comparing the prediction accuracy of LSTM, GRU, and TCN in the dataset of simulated trajectories

## Dataset
code the algorithm to generate the dataset from the paper [*].
- [*] SMOOTH : A Simple Way to Model Human Mobility

## Environment
- Linux
- RTX 3080
- python 3.8


## Models
### LSTM
Long Short-Term Memory
```
python train_and_test.py --model LSTM
```

### GRU
Gated Recurrent Unit
```
python train_and_test.py --model GRU
```

### TCN
Temporal Convolutional Network[*]
```
python train_and_test.py --model TCN
```
- [*] An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
