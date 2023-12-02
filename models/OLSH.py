from .BaseModel import BaseModel,H
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd

class OLSH(BaseModel):
    def __init__(self,args):
        # Initialize the Huber Regressor with the desired epsilon
        self.epsilon = args.epsilon 
        self.model = HuberRegressor(epsilon=self.epsilon)
        self.cols = None
        self.coefficients = None
        self.args = args

    def fit(self, train_data, train_labels):
        # Fit the Huber Regressor to the training data
        self.cols = train_data.columns
        self.model.fit(train_data, train_labels)
        self.coefficients = self.model.coef_

    def predict(self, data):
        # Predict using the Huber Regressor model
        data = data[self.cols]
        return self.model.predict(data)
    
    def get_coefficients(self):
        # Return a DataFrame with feature names and their corresponding coefficients
        # Note that we are not including the intercept in the coefficients
        coefficients_dict = dict(zip(self.cols, self.coefficients))
        df_importance = pd.DataFrame.from_dict(coefficients_dict, orient='index', columns=[self.args.model])
        return df_importance
