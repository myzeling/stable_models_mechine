import feather
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class DataUtil:
    
    def __init__(self, args, end_date='2023-03-31'):
        self.args = args
        self.df = feather.read_dataframe(self.args.path_factors)
        self.path_macros = self.args.path_macros
        self._filter_date(end_date)
        self._compute_ret_next()
        self._convert_to_datetime()
        self._merge_with_macros()

    def _filter_date(self, end_date):
        self.df = self.df[self.df['date'] <= end_date]

    def _compute_ret_next(self):
        self.df['ret_next'] = self.df.groupby('id', group_keys=False)['ret_c2c'].shift(-1)

    def _convert_to_datetime(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        macros = pd.read_csv(self.path_macros)
        macros['date'] = pd.to_datetime(macros['date'])
        self.macros = macros

    def _merge_with_macros(self):
        self.df = pd.merge(self.df, self.macros, on='date', how='left')

    def data_split(self, date):
        
        train = self.df.loc[self.df['date'] <= date, :]
        test = self.df.loc[self.df['date'] > date, :]
        return train, test

    def process_data(self,data, train=True, save=False):
        if 'industry_factor' in data.columns:
            del data['industry_factor']
        if self.args.model == 'OLS3':
            factors = ['size','bm_factor','mom6m_factor']
        else:
            factors = [i for i in list(data) if 'factor' in i]+['size','volume','amount']
        print("因子个数是:",len(factors))
        print("因子名称是:",factors)
        max_len = len(data['date'].unique())

        if train:
            stock_selected = data.groupby('id').filter(lambda x: len(x) >= 9)
            repeat_times = (max_len // stock_selected.groupby('id').size()).clip(lower=1)
            data = pd.concat([stock_selected.loc[stock_selected['id'] == id].iloc[:(max_len % len(group)) if repeat_times[id] >= 2 else len(group)].copy() 
                            for id, group in stock_selected.groupby('id') 
                            for _ in range(repeat_times[id])], 
                            axis=0, ignore_index=True)
            data.sort_values(by=["date", "id"], inplace=True)

        data.replace([np.inf, -np.inf], 0.001, inplace=True)
        data.fillna(0, inplace=True)
        cols = factors+['date','id','ret_next']
        if save:
            feather.write_dataframe(data, f'./processed_data_{"train" if train else "test"}.feather')

        return data[cols], factors
    def ultiscaler(self):
        train,test = self.data_split(self.args.split_date)
        train,factors = self.process_data(data = train,train=True,save=False)
        test,factors = self.process_data(data = test,train=False,save=False)
        df_new = pd.concat([train,test],axis=0,ignore_index=True)
        df_new.loc[:,factors] = StandardScaler().fit_transform(df_new.loc[:,factors])
        train = df_new.loc[df_new['date'] <= self.args.split_date,:]
        test = df_new.loc[df_new['date'] > self.args.split_date,:]
        if self.args.model == 'OLS3':
            cols = ['size','bm_factor','mom6m_factor']
        else:
            cols = list(train.columns)
            cols.remove('date')
            cols.remove('id')
            cols.remove(self.args.target)
        train.fillna(0.0001,inplace=True)
        test.fillna(0.0001,inplace=True)
        train.replace([np.inf, -np.inf], 0.001, inplace=True)
        test.replace([np.inf, -np.inf], 0.001, inplace=True)
        return train, test, cols