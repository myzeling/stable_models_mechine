from .BaseModel import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

class OLS(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = LinearRegression()
        self.cols = None
    def fit(self, train_data, train_labels):
        self.cols = train_data.columns
        self.model.fit(train_data, train_labels)

    def get_coefficients(self):
        coefficients = dict(zip(self.cols, self.model.coef_))
        df_importance = pd.DataFrame.from_dict(coefficients, orient='index', columns=[self.args.model])
        return df_importance

    def predict(self, data):
        return self.model.predict(data)
