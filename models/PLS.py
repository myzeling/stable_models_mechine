from .BaseModel import BaseModel
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np

class PLS(BaseModel):
    def fit(self, train_data, train_labels):
        self.model = PLSRegression(n_components=10)  # 设置你需要的组件数
        self.model.fit(train_data, train_labels)
        self.cols = train_data.columns
    def get_coefficients(self):
        coefficients = self.model.coef_
        df_importance = pd.DataFrame(coefficients, index=self.cols, columns=[self.args.model])
        return df_importance
    def predict(self, data):
        return self.model.predict(data)