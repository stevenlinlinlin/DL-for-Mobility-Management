# Deep Learning based Intelligent Connectivity for Mobility Management
使用Python Pytorch設計machine learning models預測軌跡位置  

## 目的
探討如何使用不同深度學習(Deep learning)的技術   
預測使用者接下來的位置  
並提前準備換手(Handover)的前置作業  
解決傳統上基地台被動進行換手所導致的服務品質下降及能量損耗  
比較LSTM、GRU、TCN在模擬軌跡的資料集中預測的準確度  

## 資料來源
根據論文  
SMOOTH: A Simple Way to Model Human Mobility  
模擬其演算法產生data訓練

## LSTM
LSTM(Long Short-Term Memory)
```
LSTM_pytorch.ipynb
```

## GRU
GRU(Gated Recurrent Unit)
```
GRU_pytorch.ipynb
```

## TCN
TCN(Temporal Convolutional Network)
```
TCN_pytorch.ipynb
```
