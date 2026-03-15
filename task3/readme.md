# Grid Line Usage prediction
This project builds a machine learning model to predict device power consumption (x2) using hourly sensor measurements.
The pipeline performs feature engineering, model training with XGBoost, and submission generation.

# File preprocessing
In sandbox.ipnyb at the top there is code that loads full 10gb data and sample it to have average by hour. Switching from every 5 min -> hourly reduces data size 12x making it viable to use.

# Time features
Time features are sine encoded ensuring the model can understand circular nature of those values.

# XGBoost
XGBoost was our final model of choice