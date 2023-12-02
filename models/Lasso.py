from .BaseModel import BaseModel
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.datasets import make_regression
import pandas as pd

class Lasso(BaseModel):
    def __init__(self,args):
        self.args = args
        self.cols = None
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """
        假设train和test都是包含特征和目标的数据框，且目标列名为'target'
        """
        params = {
            'alpha': 0.01
        }
        self.model = SklearnLasso(**params)
        self.model.fit(X_train, y_train)
        self.cols = X_train.columns
    def get_coefficients(self):
        """
        Lasso模型本身不提供feature_importance_属性，但我们可以使用模型的coef_属性
        作为特征的重要性。coef_表示每个特征对于目标的贡献。
        """
        importance = dict(zip(self.cols, self.model.coef_))
        df_new = pd.DataFrame.from_dict(importance,orient='index', columns=[self.args.model])
        print('look at the df_new',df_new)
        return df_new
    def predict(self, data):
        return self.model.predict(data)