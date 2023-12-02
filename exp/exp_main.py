from exp.exp_basic import Exp_Basic
from models import OLS, OLS3, PLS, Enet, XGB, LGB,OLSH,RF,Lasso
from utils.data_processor import DataUtil
from utils.Roos import PerformanceMetrics

import utils.plot as uplot
import xgboost as xgb
import lightgbm as lgb
import os
import pickle
import torch
import feather
import numpy as np
import pandas as pd
import tqdm as tqdm

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.target = args.target
        self.importance = None
        self.model_save_path = None
        # 创建一个结果文件夹
        if os.path.exists(self.args.result_path) == False:
            self.args.result_path = './results'
        net_value_path = os.path.join(self.args.result_path, 'net_value')
        # 放一个净值文件夹
        if os.path.exists(net_value_path) == False:
            os.makedirs(net_value_path)
        # 创一个模型文件夹
        if os.path.exists(self.args.checkpoints) == False:
            os.makedirs(self.args.checkpoints)  
    def save_importance_to_csv(self):
        df_new = self.importance
        if os.path.exists(self.args.path_importance):
            df_existing = pd.read_csv(self.args.path_importance, index_col=0)
            if self.args.model not in df_existing.columns:
                # 获取所有特征名称的并集
                all_features = set(df_existing.index).union(set(df_new.index))

                # 确保df_existing和df_new都有相同的特征，缺失的填充为0
                df_existing = df_existing.reindex(all_features).fillna(0)
                df_new = df_new.reindex(all_features).fillna(0)

                # 合并两个DataFrame
                df_importance = pd.concat([df_existing, df_new], axis=1)
            else:
                df_importance = df_existing
        else:
            df_importance = df_new

        # 保存合并后的结果到CSV文件
        df_importance.to_csv(self.args.path_importance, index=True, float_format='%.6f')
        return df_importance
    
    def _build_model(self):
        print("开始构建模型...")
        model_dict = {
            'OLS': OLS.OLS,
            'OLS3': OLS.OLS,
            'OLSH': OLSH.OLSH,
            'PLS': PLS.PLS,
            'Enet':Enet.Enet,
            'XGB':XGB.XGB,
            'LGB':LGB.LGB,
            'RF':RF.RF,
            'Lasso':Lasso.Lasso,
        }
        model = model_dict[self.args.model](self.args)
        return model
    
    def _get_data(self):
        data_processor = DataUtil(self.args)
        train,test,cols = data_processor.ultiscaler()
        return train,test,cols
    
    def train(self):
        print("开始训练，获取数据中...")
        train, test, cols = self._get_data()
        assert not train.isin([np.inf, -np.inf, np.nan]).any().any(), "Data contains NaN or infinite values!"
        assert not test.isin([np.inf, -np.inf, np.nan]).any().any(), "Data contains NaN or infinite values!"
        model = self._build_model()
        if self.args.model == 'XGB':
            dtrain = xgb.DMatrix(train.loc[:,cols],label = train.loc[:,self.target])
            dtest = xgb.DMatrix(test.loc[:,cols],label = test.loc[:,self.target])
            print(f"开始训练模型{self.args.model}...")
            model.grid_search(dtrain,dtest)
            y_pred = model.predict(dtest)
            print("模型训练完成，获取因子重要性中")
            self.importance = model.get_feature_importance()
        elif self.args.model == 'LGB':
            dtrain = lgb.Dataset(train.loc[:,cols],label = train.loc[:,self.target])
            dtest = lgb.Dataset(test.loc[:,cols],label = test.loc[:,self.target])
            print(f"开始训练模型{self.args.model}...")
            model.fit(dtrain, dtest)
            # 注意lgb方法里面在预测的时候不要用lgbDataset
            y_pred = model.predict(test.loc[:,cols])
            print("模型训练完成，获取因子重要性中")
            self.importance = model.get_feature_importance()
        elif self.args.model == 'RF':
            dtrain = train.loc[:,cols]
            dtest = test.loc[:,cols]
            train_label = train.loc[:,self.target]
            test_label = test.loc[:,self.target]
            print(f"开始训练模型{self.args.model}...")
            model.fit(dtrain,train_label,test_label,dtest)
            y_pred = model.predict(dtest)
            print("模型训练完成，获取因子重要性中")
            self.importance = model.get_coefficients()
        else:
            dtrain = train.loc[:,cols]
            dtest = test.loc[:,cols]
            train_label = train.loc[:,self.target]
            print(f"开始训练模型{self.args.model}...")
            model.fit(dtrain,train_label)
            y_pred = model.predict(dtest)
            print("模型训练完成，获取因子重要性中")
            self.importance = model.get_coefficients()
        self.model_save_path = os.path.join(self.args.checkpoints, f'{self.args.model}.pkl')
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model, f)
        test['pred'] = y_pred
        # 完成Roos的计算工作（每个模型来一个）
        print("------------------开始计算Roos------------------")
        PerformanceMetrics.calculate_Roos(test,self.args)
        # 完成因子重要性的统计工作
        print("------------------开始画因子重要性图-----------------")
        all_importance = self.save_importance_to_csv()
        # 我们这里还需要画一个因子重要性的图。这个功能我们交给plot去做，我觉得是ok的。
        uplot.plot_importance(all_importance,self.args.path_heatmap,self.args.path_importance_scoring)
        # 净值回撤
        uplot.PortfolioBacktest(test,self.args)

    def getformers(self):
        former_files = os.listdir(self.args.path_formers)
        for file in former_files:
            if self.args.model in file:
                selected_file  = file
        print("这次选中的模型是",self.args.model)
        print("选中的文件有",selected_file)
        for file in os.listdir(os.path.join(self.args.path_formers,selected_file)):
            if file.endswith('feather'):
                df = feather.read_dataframe(os.path.join(self.args.path_formers,selected_file,file))
                df['year'] = df['year'].astype(int)
                df['month'] = df['month'].astype(int)
                df['day'] = df['day'].astype(int)
                df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
                df.rename(columns={'stock': 'id'}, inplace=True)
                df.drop(['year', 'month', 'day'], axis=1, inplace=True)
                df = df.groupby(['id', 'date']).agg({'pred': 'mean'}).reset_index()
                orignal_data = feather.read_dataframe(self.args.path_factors)
                with open('id_mapping.pkl', 'rb') as f:
                    id_mapping = pickle.load(f)
                df['id'] = df['id'].map(id_mapping)
                df = pd.merge(df,orignal_data,on=['id','date'],how='left')
                print("看看读取的文件咋就错了\n",df)
                print("------------------开始计算Roos------------------")
                PerformanceMetrics.calculate_Roos(df,self.args)
                uplot.PortfolioBacktest(df,self.args)
            else:
                self.importance = pd.read_csv(os.path.join(self.args.path_formers,selected_file,file),index_col=0)
                print("看看读取的文件咋就错了\n",self.importance)
                self.importance = self.importance.astype(float)
                all_importance = self.save_importance_to_csv()
                print("------------------开始画因子重要性图-----------------")
                uplot.plot_importance(all_importance,self.args.path_heatmap)

    # def test(self):
    #     # 检查模型是否已存在
    #     if os.path.exists(self.model_save_path):
    #         print(f"加载已存在的模型：{self.args.model}...")
    #         with open(self.model_save_path, 'rb') as f:
    #             model = pickle.load(f)
    #         if self.args.model == 'XGB':
    #             dtrain = xgb.DMatrix(train.loc[:,cols],label = train.loc[:,self.target])
    #             dtest = xgb.DMatrix(test.loc[:,cols],label = test.loc[:,self.target])
    #             print(f"开始训练模型{self.args.model}...")
    #             model.fit(dtrain,dtest)
    #             y_pred = model.predict(dtest)
    #             print("模型训练完成，获取因子重要性中")
    #             self.importance = model.get_feature_importance()
    #         elif self.model == 'LGB':
    #             dtrain = lgb.Dataset(train.loc[:,cols],label = train.loc[:,self.target])
    #             dtest = lgb.Dataset(test.loc[:,cols],label = test.loc[:,self.target])
    #             print(f"开始训练模型{self.args.model}...")
    #             model.fit(dtrain, dtest)
    #             # 注意lgb方法里面在预测的时候不要用lgbDataset
    #             y_pred = model.predict(test.loc[:,cols])
    #             print("模型训练完成，获取因子重要性中")
    #             self.importance = model.get_feature_importance()
    #         else:
    #             dtrain = train.loc[:,cols]
    #             dtest = test.loc[:,cols]
    #             train_label = train.loc[:,self.target]
    #             print(f"开始训练模型{self.args.model}...")
    #             model.fit(dtrain,train_label)
    #             y_pred = model.predict(dtest)
    #             print("模型训练完成，获取因子重要性中")
    #             self.importance = model.get_coefficients()
    #     else:
    #         print("您还未训练模型，请先训练该模型再开始test模式...")

    #     # ...[其余的代码不变]