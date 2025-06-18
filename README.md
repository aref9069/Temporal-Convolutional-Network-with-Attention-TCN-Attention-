# Temporal-Convolutional-Network-with-Attention-TCN-Attention-
This repository provides a full pipeline for Remaining Useful Life (RUL) prediction using the NASA C-MAPSS dataset. It includes structured preprocessing, advanced feature engineering, a Temporal Convolutional Network with Attention (TCN-Attention) model, and robust evaluation and visualization tools.


📦 Features

✅ Clean ingestion and parsing of raw C-MAPSS files  
✅ Normalization grouped by operating conditions  
✅ Capped RUL labeling using piecewise linear rules  
✅ Cumulative feature engineering for multi-condition datasets  
✅ Sliding-window sequence generation for sequence-to-one prediction  
✅ Model support: **BLSTM-CNN with Attention** (can be extended)  
✅ Evaluation metrics: RMSE, MAE, R², NASA Exponential Score  
✅ Automatic visualizations of predictions and scatter plots  
✅ CSV export of model performance (RMSE, Score)


⚙️ Requirements

pip install numpy pandas scikit-learn tensorflow matplotlib
