from .BaseModel import BaseModel,H
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
class PLSH(BaseModel):
    def __init__(self, n_components=2, max_iter=500):
        self.pls = PLSRegression(n_components=n_components, max_iter=max_iter)
        self.coefficients = None
        self.cols = None
        self.xi = self.args.xi

    def fit(self, train_data, train_labels):
        self.cols = train_data.columns
        self.pls.fit(train_data, train_labels)
        # 使用H函数调整系数
        self.coefficients = np.array([H(coef[0], self.xi) for coef in self.pls.coef_])

    def predict(self, data):
        return data @ self.coefficients
    
    def get_coefficients(self):
        # Assuming that 'train_data' columns are the feature names
        feature_names = self.cols
        coefficients_dict = dict(zip(feature_names, self.coefficients))
        df_importance = pd.DataFrame.from_dict(coefficients_dict, orient='index', columns=[self.args.model])
        return df_importance