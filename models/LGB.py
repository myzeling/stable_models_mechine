from .BaseModel import BaseModel
import lightgbm as lgb
import pandas as pd

class LGB(BaseModel):
    def fit(self,train,test):
        params = {
            'objective': 'regression',
            'num_leaves': 101,
            'learning_rate': 0.001,
            'n_estimators': 111,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_child_samples': 20,
        }
        num_round = 389
        self.model = lgb.train(params, train, num_round, valid_sets=[test], early_stopping_rounds=10)
    def get_feature_importance(self, importance_type='split'):
        """
        Get feature importances.
        
        Parameters:
        - importance_type (str): How the importance is calculated. Can be "split" or "gain".
                                 "split" is the number of times a feature is used in models.
                                 "gain" is the total gain of splits which use the feature.
        """
        importance_scores = self.model.feature_importance(importance_type=importance_type)
        feature_names = self.model.feature_name()
        df_new = pd.DataFrame(importance_scores, index=feature_names, columns=[self.args.model])
        return df_new
    def predict(self, data):
        return self.model.predict(data)