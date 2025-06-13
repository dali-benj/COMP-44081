# Technical Test for Applied Scientist (Data Scientist)

__Opportunity No. 44081__

## Project Summary
The goal is to improve the provided model to forecast ticket sales and redemptions. The original model was too simple, as it did not take into account factors from the real-world that influence travel decisions.
We have built a smarter and more accurate model. Here's how:
- Understand the "Why": We added demand drivers such as weather information (temperature, rain, snow, wind), public holidays, and seasonality.
- Sanitized the data: We removed the Covid years (2020,2021) from our training data as these are clear outliers and are likely to throw our model off. We want our model to reflect typical travel patterns, not a once-in-a-lifetime anomaly.
- Compared approaches: We tested different models to see which one would perform better. The XGBoost model showed the best performance compared to the others.
- Delivered results: The final model provides daily forecasts for ticket sales and redemptions that are 50% more accurate than the base model, which enables better planning.

## Technical description
To enhance the existing base model, we focused on comprehensive feature engineering, outlier handling, and implementation of machine learning models.
1. Feature engineering
The main improvement was the feature engineering process. We augmented the original dataset with several features:
- Time-based features: day_of_week, day_of_year, month, year were created to capture cyclical patterns.
- Holiday effects: Using the holiday library, we created a binary feature to flag all Ontario holidays and model the accompanying lift in sales.
- Weather data: Using the meteostat library, we fetched historical weather data for Toronto. We used information and min/max temperatures, wind speed, snowfall, and rainfall. 
- Lag feature: We used a 365-day lagged feature to provide a year-over-year signal.
2. Modeling and Evaluation
A key decision was to remove the data from 2020 and 2021 from the training dataset. This prevents skewed data from the pandemic period from affecting the training process. To solve the forecasting problem, we compared two distinct model against the baseline within a cross-validation framework.
- FB Prophet: We implemented it for its powerful and automatic handling of various seasonality and holiday effects.
- XGBoost: This regressor was used for its capacity to leverage the engineered feature set and capture patterns and complex non-linear relationships between input variables. 
- Deep Learning Models (LSTM, RNN, etc): Could have been a choice for this problem, but we decided against it as there is no incentive to go for a complicated model when the input data size is very small.

3. Model Performance
The models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) accross four cross-validation splits. Mean Absolute Percentage Error was calculated, but it is unreliable due to zeros in the test data.
- XGBoost model: This was the best performing model, and achieved the lowest MAE (1213) and RMSE (2133). Its ability to capture complex relationships between different input variables proved useful in improving the forecast.
- Prophet: Showed a good improvement over the base model, but was not as accurate as the gradient boosting model.

Based on this quantitative analysis, the XGBoost model is recommended. Given that the achieved MAE is still high, we may try some other approaches to further improve the model. 
An example would be to aggregate the forecast at a weekly level to make it easier to predict, and then spread down the weekly prediction to the day level according to a weekly spread profile.


## Use of AI
AI (Google Gemini) was used during the development process as a tool for brainstorming. 
