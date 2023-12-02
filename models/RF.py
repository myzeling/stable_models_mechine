from .BaseModel import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd

class RF(BaseModel):
    def fit(self, train,train_label,test_label, test):
        params = {
            'n_estimators': 100,  # 树的数量
            'max_depth': None,    # 树的最大深度
            'min_samples_split': 2,  # 分割内部节点所需的最小样本数
            'random_state': 42    # 随机数种子
        }
        self.model = RandomForestRegressor(**params)
        self.model.fit(train, train_label)
        test_predictions = self.model.predict(test)
        test_loss = mean_squared_error(test_label, test_predictions)
        print(f"Test Loss: {test_loss}")
        self.cols = train.columns

    def get_feature_importance(self):
        importance = self.model.feature_importances_
        df_new = pd.DataFrame(importance, index=self.cols, columns=[self.args.model])
        return df_new

    def predict(self, data):
        return self.model.predict(data.loc[:, data.columns != self.args.target])

    def grid_search(self, train, train_label, cv=3, scoring='neg_mean_squared_error'):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=cv, scoring=scoring, verbose=3)
        grid_search.fit(train, train_label)
        self.model = grid_search.best_estimator_
