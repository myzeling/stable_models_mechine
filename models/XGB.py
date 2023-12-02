from .BaseModel import BaseModel
import xgboost as xgb
import pandas as pd
class XGB(BaseModel):
    def fit(self, train, test):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'eta': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': '42'
        }
        self.model = xgb.train(params, train, num_boost_round=10000, evals=[(test, 'test')],
                               early_stopping_rounds=100, verbose_eval=10)
    def get_feature_importance(self):
        # 默认的重要性类型是 'weight'，您也可以选择 'gain' 或 'cover'
        '''
        Weight (或 Count):
        这表示特定特征用于构建模型中树的次数。例如，如果某个特征在所有的树中总共被用了5次作为一个分割点，那么它的重要性权重就是5。
        通常，重要性权重可以告诉我们这个特征被模型使用的频率，但它不告诉我们使用这个特征带来了多大的性能提升。
        Gain (或 Split Mean Gain):
        表示每次特征用于分割时，它带来的平均增益。这个增益是指在特征分割点的两侧的不纯度的减少，常用的不纯度指标包括基尼不纯度、均方误差等。
        这个指标提供了更深入的信息，显示了该特征对模型性能的改进程度。
        Cover (或 Split Mean Coverage):
        表示每次特征用于分割时，它影响到的样本数的平均值。在分类任务中，这表示每次使用该特征进行分割时的平均样本数；在回归任务中，这表示在每次分割点的两侧的样本权重的总和。
        这个指标可以帮助我们理解这个特征对哪些样本起作用。
        为了选择哪种特征重要性度量方法最合适，要根据特定的应用和你想知道的信息进行选择。例如，如果你想知道哪些特征最多次被模型使用，选择weight；如果想知道哪些特征带来了最大的性能提升，选择gain；如果你想知道哪些特征对哪些样本起作用，选择cover。
        '''
        importance = self.model.get_score(importance_type='weight')
        df_new = pd.DataFrame.from_dict(importance, orient='index', columns=[self.args.model])
        return df_new
    def predict(self, data):
        return self.model.predict(data)
    def grid_search(self, train, test, nfold=3, num_boost_round=100, early_stopping_rounds=50):
        param_grid = [
            {'max_depth': 3, 'eta': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'max_depth': 4, 'eta': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8},
            # 添加更多参数组合
        ]
        min_rmse = float("Inf")
        best_params = None

        # 进行网格搜索找到最佳参数
        for param in param_grid:
            cv_results = xgb.cv(param, train, num_boost_round=num_boost_round,
                                nfold=nfold, early_stopping_rounds=early_stopping_rounds, 
                                metrics='rmse', as_pandas=True, seed=42)
            mean_rmse = cv_results['test-rmse-mean'].min()
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_params = param

        print(f"Best Params: {best_params}")
        print(f"Minimum RMSE: {min_rmse}")

        # 使用最佳参数重新训练模型
        self.model = xgb.train(best_params, train,evals=[(test, 'test')], num_boost_round=num_boost_round)

        