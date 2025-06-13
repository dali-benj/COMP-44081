import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error , mean_squared_error,mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np

import holidays
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import xgboost as xgb
from prophet import Prophet


class RedemptionModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # dict of dicts with model results

    def score(self, truth, preds):
        # Score our predictions - modify this method as you like
        return [mean_absolute_percentage_error(truth, preds),mean_absolute_error(truth,preds),np.sqrt(mean_squared_error(truth, preds))]


    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        self.X = self.X[~self.X.index.year.isin([2020, 2021])] #Removing covid years

        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]

            # Base model - please leave this here
            preds = self._base_model(X_train, X_test)
            if 'Base' not in self.results:
                self.results['Base'] = {}
            self.results['Base'][cnt] = self.score(X_test[self.target_col],preds)
            self.plot(preds, 'Base')
            # XGBoost Model
            preds_xgb= self._xgboost_model(X_train, X_test)
            
            if 'XGBoost' not in self.results: self.results['XGBoost'] = {}
            self.results['XGBoost'][cnt] = self.score(X_test[self.target_col], preds_xgb)
            self.plot(preds_xgb, 'XGBoost Model')
            # Prophet Model
            preds_prophet = self._prophet_model(X_train, X_test)
            if 'Prophet' not in self.results: self.results['Prophet'] = {}
            self.results['Prophet'][cnt] = self.score(X_test[self.target_col], preds_prophet)
            self.plot(preds_prophet, f'Prophet Model')

            # Other models...
            # self._my-new-model(train, test) << Add your model(s) here
            cnt += 1

    def _engineer_features_for_ml(self, train, test):
        
        train_featured = train.copy()
        test_featured = test.copy()
        combined = pd.concat([train_featured, test_featured])

        combined['day_of_week'] = combined.index.dayofweek
        combined['day_of_year'] = combined.index.dayofyear
        combined['month'] = combined.index.month
        combined['year'] = combined.index.year
        combined['is_holiday'] = combined.index.map(lambda x: 1 if x in holidays.Canada(prov='ON') else 0)

        combined['lag_365'] = combined[self.target_col].shift(365)
        
        start_year = datetime(combined['year'].min(), 1, 1)
        end_year = datetime(combined['year'].max(), 12, 31)

        location = Point(48.6333, -79.45, 269.0)
        weather_df = Daily(location, start_year, end_year)
        weather_df = weather_df.fetch()
        #print(weather_df)
        weather_df['tmin']=pd.to_numeric(weather_df['tmin'], errors='coerce')
        weather_df['tmax']=pd.to_numeric(weather_df['tmax'], errors='coerce')
        weather_df['prcp']=pd.to_numeric(weather_df['prcp'], errors='coerce')
        weather_df['snow']=pd.to_numeric(weather_df['snow'], errors='coerce')
        weather_df['wspd']=pd.to_numeric(weather_df['wspd'], errors='coerce')
        combined= combined.join(weather_df[['tmin','tmax','prcp','snow','wspd']]).fillna(method='ffill').fillna(method='bfill')
        train_final = combined.loc[train.index]
        test_final = combined.loc[test.index]
        
        return train_final, test_final
    
    def _xgboost_model(self, train, test):
        train_features, test_features = self._engineer_features_for_ml(train, test)

        features = [col for col in train_features.columns if (col != self.target_col and col!="Sales Count" and col!="Redemption Count" and col!="_id") ]
        
        X_train, y_train = train_features[features], train_features[self.target_col]
        X_test, y_test = test_features[features], test_features[self.target_col]

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=132,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        preds[preds < 0] = 0

        feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print("\nXGBoost Feature Importances:")
        print(feature_importance)
        print("-" * 30)
        
        return pd.Series(preds, index=test.index)

    def _prophet_model(self, train, test):
        
        train_prophet = train.reset_index().rename(columns={'Timestamp': 'ds', self.target_col: 'y'})
        ontario_holidays = holidays.Canada(prov='ON', years=list(range(train.index.min().year, test.index.max().year + 1)))
        holidays_df = pd.DataFrame(list(ontario_holidays.items()), columns=['ds', 'holiday'])
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        
        model = Prophet(holidays=holidays_df)
        model.fit(train_prophet)
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        preds = forecast.iloc[-len(test):]['yhat']
        preds[preds < 0] = 0
        preds.index = test.index
        return preds

    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col],
                                        period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index = test.index, 
                         data = map(lambda x: res_dict[x], test.index.dayofyear))

    def plot(self, preds, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label, color='red')
        plt.legend()


