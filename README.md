# STOCK-PREDICTION
The aim of this project is to develop a forecasting model, for Apple Inc. (AAPL) stock, using deep learning techniques. Here, we have applied LSTM networks to forecast the closing price and classify the next-day movement as either upward or downward. 
The model combines the use of data preprocessing, technical indicator engineering, sequence modelling, and deep learning to deliver predictions and trading signals that would be useful.
OBJECTIVE
•	To predict next day's closing price of AAPL using an LSTM regression model.
•	To predict the direction (up or down) of next-day movement of the stock prices using a Bidirectional LSTM classifier.
•	To generate trading signals (Buy/Sell/Hold) based on model predictions and RSI values
TOOLS AND TECHNOLOGIES USED
•	Python (Libraries: Pandas, Numpy, Scikit-learn, Keras, TensorFlow, and Matplotlib)
•	Yfinance (Yahoo Finance) for financial data
•	ta library for technical indicators
•	Keras Tuner for hyperparameter tuning
•	Google Colab environment

METHODOLOGY
Data Collection and Feature Engineering
Data for AAPL stock (2015–2023) was collected using the yfinance API. The dataset included OHLC values and the volume of stocks traded. The technical features derived through the dataset were:
•	RSI (Relative Strength Index)
•	MACD (Moving Average Convergence Divergence)
•	SMA-20 (Simple Moving Average)
•	1-day Return
•	Close Lag (previous day’s close)

Data Preprocessing
•	Feature normalization was done using MinMaxScaler.
•	For each prediction, data was framed as time-series sequences of 100 previous days.
•	Target variables were: (i) next-day price (regression)
(ii) next-day direction (0/1 classification).
Model Design
Two separate models were built and trained for the project:
•	Price Prediction Model: LSTM → Dropout layer → LSTM → Dense layer
•	Direction Prediction Model: Bidirectional LSTM → Dropout layer → Dense layer (using sigmoid)
•	Both models were tuned using keras_tuner.RandomSearch
ARCHITECTURE
Price Prediction Model (LSTM)
Input: 100-day sequence of scaled features
Architecture Layers:
•	LSTM layer with tuned units
•	Dropout (to prevent overfitting)
•	Another LSTM layer
•	Final Dense layer (1 neuron, linear activation)
Loss Function: Mean Squared Error
Optimizer: Adam
Direction Prediction Model (Bidirectional LSTM)
Input: Same 100-day sequence
Architecture Layers:
•	Bidirectional LSTM layer
•	Dropout
•	Dense layer with sigmoid activation (binary classification)
•	Loss Function: Binary Crossentropy
Metrics: Accuracy, Precision, Recall, F1 Score

EVALUATION
Result interpretation and analysis
Price Prediction:
•	The Actual vs Predicted Price plot shows that the model closely follows the real stock movements over time. It captures both trends and reversals almost accurately.
•	The Error Distribution Histogram indicates that most errors lie in the range of -5 to +5 USD. Errors are following a roughly normal distribution centred around 0, which is a positive sign.
•	Bar graph metrics:
o	MSE: 21.36
o	MAE: 3.62
o	RMSE: 4.62
o	R² Score: 0.95 (Very high — indicates 95% of price variance is explained by the model)
•	Conclusion: Price prediction model is accurate and well-calibrated for short-term predictions.

Direction Prediction:
•	The Confusion Matrix shows the model predicted:
o	311 true positives (1 predicted as 1)
o	81 true negatives
o	278 false positives
o	79 false negatives
•	Metric scores:
o	Accuracy: 52%
o	Precision: 53%
o	Recall: 80% 
o	F1 Score: 64%
Conclusion: While the accuracy of the model is moderate (52%), it is very sensitive to "up" movements (recall = 80%), which would be valuable in trading contexts.

LIMITATIONS
•	Feature Limitation: Only technical indicators were used to train both models. No news, volume sentiment, or macroeconomic factors were used.
•	Short-Term Focus: The model predicts only one day ahead which makes it not completely reliable longer forecasts.
•	Overfitting Possibility: Training vs validation loss (seen in direction model) suggests some instability or overfitting. It could be improved with regularization or using vast training data.
•	Unbalanced Decisions: Confusion matrix shows a bias towards predicting upward movement. This could negatively impact trading by incurring losses.

FUTURE ENHANCEMENTS
•	Incorporating News/Sentiment Analysis: Adding Twitter, news feeds, or analyst reports can help capture event-driven volatility.
•	Multistep Forecasting: Predicting 3, 5, or 10 days ahead using encoder-decoder LSTMs or attention mechanisms.
•	Use Transformer Models: For better long-sequence handling and trend detection.
•	Backtesting Strategy: Implement real trades on historical data to test profits/losses from model signals.
•	Optimize RSI Signal Logic: Use thresholds dynamically rather than static 30/70 cutoffs.
