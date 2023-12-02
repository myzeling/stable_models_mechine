import pandas as pd
import feather
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class PortfolioBacktest:
    def __init__(self, df, args):
        self.df = df.copy()
        self.models = ['XGB', 'LGB', 'OLS', 'OLS3', 'Enet','OLSH','PLS','PLSH','Transformer','Informer','Autoformer']
        self.args = args
        self.model = args.model
        self.file_path = args.path_net_value
        if os.path.exists(self.file_path):
            self.net_values_df = pd.read_feather(self.file_path)
        else:
            self.net_values_df = pd.DataFrame()
        self.run_backtest()
        self.plot_net_values()

    def _select_weights(self, group_df, col):
        """
        Compute the weights for the top 10% and bottom 10% of stocks based on the forecast.
        """
        num_pos = int(self.args.per * len(group_df))
        # num_pos = 10
        group_df = group_df.sort_values(by=col)
        group_df[col + '_long_weight'] = 0
        group_df[col + '_short_weight'] = 0
        if num_pos <= 20:
            num_pos = len(group_df)
            group_df.iloc[-num_pos:, group_df.columns.get_loc(col + '_long_weight')] = 1 / num_pos
        else:
            group_df.iloc[:num_pos, group_df.columns.get_loc(col + '_short_weight')] = -1 / num_pos
            group_df.iloc[-num_pos:, group_df.columns.get_loc(col + '_long_weight')] = 1 / num_pos
        # group_df.iloc[:num_pos, group_df.columns.get_loc(col + '_short_weight')] = -1 / num_pos
        # group_df.iloc[-num_pos:, group_df.columns.get_loc(col + '_long_weight')] = 1 / num_pos
        return group_df

    def long_short(self):
        self.df = self.df.groupby('date',as_index = False).apply(self._select_weights, col='pred')
        self.df['long_return'] = self.df['pred_long_weight'] * self.df[self.args.target]
        self.df['short_return'] = self.df['pred_short_weight'] * self.df[self.args.target]
        df_long = self.df.groupby('date',as_index = False)['long_return'].sum()
        df_long['type'] = 'long'
        df_long[self.model] = (1.0+df_long['long_return']).cumprod()
        # print('let us see what is wrong with the df_long',df_long)
        max_drawdown_long = ((df_long[self.model].cummax() - df_long[self.model]) / df_long[self.model].cummax()).max()
        df_short = self.df.groupby('date',as_index = False)['short_return'].sum()
        df_short['type'] = 'short'
        df_short[self.model] = (1+df_short['short_return']).cumprod()
        # print('let us see what is wrong with the df_short',df_short)
        max_drawdown_short = ((df_short[self.model].cummax() - df_short[self.model]) / df_short[self.model].cummax()).max()
        with open(self.args.result_path + f'/max_drawdown_{self.args.per}.txt', 'a') as f:
            f.write(f'{self.model}_long:{max_drawdown_long}\n{self.model}_short:{max_drawdown_short}\n')
        net_values = pd.concat([df_long, df_short])
        return net_values


    def run_backtest(self, save=True):
        net_values = self.long_short()[['date', 'type', self.model]]
        
        # 如果模型名不在DataFrame列中，则合并数据
        # 如果模型名不在DataFrame列中，则合并数据
        if self.model not in self.net_values_df.columns:
            if 'date' in self.net_values_df.columns:
                self.net_values_df = pd.merge(self.net_values_df, net_values, on=['date', 'type'], how='outer')
            else:
                self.net_values_df = net_values
        else:
            # 如果模型名已经存在，则只更新这个模型的数据
            self.net_values_df = self.net_values_df[self.net_values_df.columns.difference([self.model], sort=False)]
            self.net_values_df = pd.merge(self.net_values_df, net_values, on=['date', 'type'], how='outer')

        # Save results if needed
        if save:
            self.net_values_df.reset_index(drop=True).to_feather(self.file_path)
    
    def plot_net_values(self):
        plt.style.use('ggplot')  # 使用ggplot风格
        plt.figure(figsize=(14, 7))

        # 定义颜色，使用Matplotlib的颜色循环
        colors = plt.cm.tab10.colors  # 使用10个不同的颜色
        cols = self.net_values_df.columns.drop(['date', 'type'])
        models = [i for i in cols if i in self.models]
        for i, model in enumerate(models):
            # 长仓净值曲线（实线）
            model_data_long = self.net_values_df[(self.net_values_df['type'] == 'long') & (self.net_values_df['date'] > self.args.split_date)]
            plt.plot(model_data_long['date'], model_data_long[model], label=f'{model}_long', color=colors[i], linestyle='-')

            # 短仓净值曲线（虚线）
            model_data_short = self.net_values_df[(self.net_values_df['type'] == 'short') & (self.net_values_df['date'] > self.args.split_date)]
            plt.plot(model_data_short['date'], model_data_short[model], label=f'{model}_short', color=colors[i], linestyle='--')

        plt.title('Net Value Curves of Different Models')
        plt.xlabel('Date')
        plt.ylabel('Net Value')
        
        # 设置日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        plt.legend()
        plt.grid(True)  # 打开网格
        plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
        plt.savefig(self.args.net_value_fig)


def normalize(df):
    df = df.abs()
    return (df - df.mean()) / df.std()

def plot_importance(df,path_heatmap,path_importance_scoring):
    df = abs(df)
    df_s = df/df.sum()*100
    df_s['Overall_Importance'] = df_s.sum(axis=1)
    df_s = df_s.sort_values(by='Overall_Importance', ascending=False)
    print("let's see what is wrong with the df_s",df_s)
    df_s.drop('Overall_Importance', axis=1, inplace=True)
    df_scoring = df_s
    df_scoring.to_csv(path_importance_scoring)
    # 使用seaborn绘制热图
    plt.figure(figsize=(12, 30))
    heatmap = sns.heatmap(df_s, cmap='Blues', linewidths=0.5, linecolor='gray')
    heatmap.set_title('Feature Importance', fontdict={'fontsize':18}, pad=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.title("Feature Importance")
    plt.savefig(path_heatmap)