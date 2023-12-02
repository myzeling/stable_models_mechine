from .BaseModel import BaseModel
import statsmodels.api as sm
import pandas as pd
class OLS3(BaseModel):
    def fit(self, train_data, train_labels):
        self.model = sm.OLS(train_labels, train_data).fit()
    def get_coefficients(self):
        coefficients = self.model.params.to_dict()
        df_importance = pd.DataFrame.from_dict(coefficients, orient='index', columns=[self.args.model])
        return df_importance
    def predict(self, data):
        return self.model.predict(data)