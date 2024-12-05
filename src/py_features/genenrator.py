import os
import re
from datetime import timedelta, datetime

import pydash
import pandas as pd
import numpy as np
from typing import List, Dict, Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FEATURE_DEFAULT_PATH = f'data/pickle'
np.seterr(divide='ignore', invalid='ignore')


class FeatureGenerator:
    """
    raw_data
    1. 将原始量价数据转换为features，
    2. features存储到本地
    """
    path = FEATURE_DEFAULT_PATH
    # 1h 4h 8h 12h 1D 2D 3D 5D 10D 20D 40D
    rolling_windows: List[int] = [1, 4, 8, 12, 24, 24 * 2, 24 * 3, 24 * 5, 24 * 10, 24 * 15, 24 * 20, 24 * 40]
    features: Dict[str, pd.DataFrame]

    def __init__(self, raw_data: pd.DataFrame, bar: str = '15m',
                 rolling_windows: List[int] = None, pair_delimiter: str = '-', path: str = None):
        """

        :param raw_data: 原始数据，需包含`ts` `trade_code` `close` `high` `low` `open` `volume`列
        :param bar: 时间周期
        :param path: features最终存储的位置
        """
        raw_data.columns = [c.lower() for c in raw_data.columns]
        self.raw_data = raw_data

        self.bar = bar
        self.period_unit = re.search('[a-zA-Z]+$', bar).group()
        self.step_per_period_unit = self.get_step_by_period_unit()
        self.features = {}
        self.pair_delimiter = pair_delimiter

        if rolling_windows is not None:
            self.rolling_windows = rolling_windows

        if path is not None:
            self.path = path

    def __call__(self, save: bool = True, save_method: str = 'replace',
                 aggregate_len: int = None, start_time: str = None, end_time: str = None, start_hour: int = 1):
        data = self.raw_data.copy().sort_values(by=['ts'])
        data['ts'] = pd.to_datetime(data['ts'])
        data = data.drop(columns=['bar', 'confirm'])
        if start_time is not None:
            data = data[data.ts >= pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S')]
        if end_time is not None:
            data = data[data.ts >= pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')]
        data = self._aggregate_raw_data(data, aggregate_len, start_hour)
        full_data = pd.MultiIndex.from_product([data.trade_code.drop_duplicates().sort_values(),
                                                data.ts.drop_duplicates().sort_values()]).to_frame().reset_index(
            drop=True)
        data = full_data.merge(data, how='left', on=['trade_code', 'ts']).sort_values(by=['trade_code', 'ts'])
        data.reset_index(drop=True, inplace=True)
        data.sort_values(['trade_code', 'ts'], inplace=True)
        data.loc[:, ['close', 'open', 'high', 'low']] = data.groupby('trade_code')[[
            'close', 'open', 'high', 'low']].ffill().reset_index(drop=True)
        data.loc[:, ['volume', 'amount']] = data[['volume', 'amount']].fillna(0)

        self.get_price_chg(data)
        data = self.get_vwap(data)
        # self.get_mru(data)
        # self.get_mdd(data)
        # self.get_ma(data)
        # self.get_ema(data)
        # self.get_volatility(data)
        # self.get_macd(data)
        # self.get_rsi(data)
        # data = self.get_volume(data)
        # data = self.get_vol_pct(data)
        # data = self.get_amt_pct(data)

        for c in data.columns:
            if c not in ['trade_code', 'ts']:
                temp = data[['trade_code', 'ts', c]].dropna()
                # temp = data[['trade_code', 'ts', c]]
                pivot_df = pd.pivot(temp, index='ts', columns=['trade_code'], values=[c])
                pivot_df.columns = [v[1] for v in pivot_df.columns]
                # pivot_df = pivot_df.fillna(0)
                self.features[c] = pivot_df

        corr_data = None
        # corr_data = self.get_corr_optimized(data)
        # for c in corr_data.columns:
        #     if c not in ['trade_code', 'ts']:
        #         pivot_df = pd.pivot(corr_data, index='ts', columns=['trade_code'], values=[c])
        #         pivot_df.columns = [v[1] for v in pivot_df.columns]
        #         self.features[c] = pivot_df

        if save:
            for n, df in self.features.items():
                file_path = f'{self.path}/{n}.pkl'
                if start_time is not None:
                    df = df[df.index >= start_time]

                if save_method == 'append':
                    prev_data = pd.read_pickle(file_path)
                    prev_data = prev_data.loc[prev_data.index < pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S'),
                                :]
                    df = pd.concat([prev_data, df])
                elif save_method == 'replace':
                    pass
                else:
                    Warning('Save method {} not implemented'.format(save_method))
                df.to_pickle(file_path)
        return data, corr_data

    def _aggregate_raw_data(self, data: pd.DataFrame, aggregate_len: int = None, start_hour: int = 1):
        copy_data = data.copy()
        if aggregate_len is not None:
            copy_data.set_index('ts', inplace=True)
            agg_methods = {
                'open': lambda s: s.iloc[0],  # open取聚合后第一条数据
                'high': 'max',  # high取聚合后最高值
                'low': 'min',  # low取聚合后最低值
                'close': lambda s: s.iloc[-1],  # close取聚合后最后一条数据
                'volume': 'sum',  # volume取总和
                'amount': 'sum',  # volume取总和
            }
            w = pd.Timedelta(self.bar).total_seconds() * aggregate_len
            # 进行时间聚合，按1小时为窗口进行聚合
            copy_data = copy_data.groupby('trade_code').rolling(aggregate_len).agg(agg_methods)
            copy_data.reset_index(inplace=True)
            copy_data['ts'] = copy_data['ts']
        return copy_data

    def get_step_by_period_unit(self):
        bar_seconds = pd.Timedelta(self.bar).total_seconds()
        period_seconds = pd.Timedelta(f'1{self.period_unit}').total_seconds()
        return int(max(period_seconds / bar_seconds, 1))

    def get_period_by_step(self, steps: int):
        source_seconds = pd.Timedelta(self.bar).total_seconds()
        total_seconds = source_seconds * steps
        if self.period_unit in ['D', 'day', 'days']:
            return f"{int(total_seconds / (60 * 60 * 24))}D"
        elif self.period_unit in ['h', 'hr', 'hour', 'hours']:
            return f"{int(total_seconds / (60 * 60))}h"
        elif self.period_unit in ['m', 'min', 'minute', 'minutes']:
            return f"{int(total_seconds / (60))}m"
        elif self.period_unit in ['S', 'sec', 'second', 'seconds']:
            return f"{int(total_seconds)}S"

    def get_price_chg(self, data: pd.DataFrame):
        data['chg'] = data.groupby('trade_code')['close'].pct_change()
        data['over_period_chg'] = data['open'] / data.groupby('trade_code')['close'].shift(1).reset_index(drop=True) - 1
        data['open_chg'] = data.groupby('trade_code')['open'].pct_change().reset_index(drop=True)
        data['high_chg'] = data.groupby('trade_code')['high'].pct_change().reset_index(drop=True)
        data['low_chg'] = data.groupby('trade_code')['low'].pct_change().reset_index(drop=True)
        data['high_begin_chg'] = data['high'] / data['open'] - 1
        data['low_begin_chg'] = data['low'] / data['open'] - 1
        data['high_end_chg'] = data['close'] / data['high'] - 1
        data['low_end_chg'] = data['close'] / data['low'] - 1
        data['amplitude'] = data['high'] / data['low'] - 1
        data['upper_shadow_ratio'] = data['high'] / data[['open', 'close']].max(axis=1) - 1
        data['lower_shadow_ratio'] = data[['open', 'close']].min(axis=1) / data['low'] - 1
        data['body_length_ratio'] = (data['open'] - data['close']).abs() / (data['high'] - data['low'])
        data['body_length_ratio'] = data['body_length_ratio'].fillna(0)
        data['volume_log'] = np.log(data['volume'])
        data['volume_log_chg'] = data.groupby('trade_code')['volume_log'].pct_change()
        data.drop('volume_log', axis=1, inplace=True)

    def _rolling_prepare(self, name: str, w: int):
        steps = w * self.step_per_period_unit
        period = self.get_period_by_step(steps)
        indic_name = f'{period.lower()}_{name.lower()}'
        return steps, indic_name

    def get_ma(self, data: pd.DataFrame):
        print('Processing MA')
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('ma', rolling_window)
            data[indic_name] = data.groupby('trade_code')['close'].rolling(int(steps)).mean().reset_index(drop=True)
            data[f'{indic_name}_chg'] = data.groupby('trade_code')[indic_name].pct_change().reset_index(drop=True)

    def get_ema(self, data: pd.DataFrame):
        print('Processing EMA')
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('ema', rolling_window)
            data[indic_name] = data.groupby('trade_code')['close'].ewm(span=steps, adjust=False).mean().reset_index(
                drop=True)
            data[f'{indic_name}_chg'] = data.groupby('trade_code')[indic_name].pct_change().reset_index(drop=True)

    def get_volatility(self, data: pd.DataFrame):
        print('Processing Volatility')
        if 'chg' not in data.columns:
            data['chg'] = data.groupby('trade_code')['close'].pct_change()

        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('volatility', rolling_window)
            data[indic_name] = data.groupby('trade_code')['chg'].rolling(int(steps)).std().reset_index(drop=True)

    def get_macd(self, data: pd.DataFrame):
        print('Processing MACD')
        data['ema12'] = data.groupby('trade_code')['close'].ewm(span=12, adjust=False).mean().reset_index(drop=True)
        data['ema26'] = data.groupby('trade_code')['close'].ewm(span=26, adjust=False).mean().reset_index(drop=True)
        data['macd'] = data['ema12'] - data['ema26']
        data['single_line'] = data.groupby('trade_code')['macd'].ewm(span=9, adjust=False).mean().reset_index(drop=True)
        data.drop(columns=['ema12', 'ema26'], inplace=True)

    def get_rsi(self, data: pd.DataFrame):
        print('Processing RSI')

        # 计算收益率变化的差分（delta）
        data['delta'] = data.groupby('trade_code')['chg'].diff()

        # 分离上涨和下跌
        data['gain'] = data['delta'].clip(lower=0)
        data['loss'] = -data['delta'].clip(upper=0)

        # 对每个滚动窗口长度进行计算
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('rsi', rolling_window)

            # 计算平均上涨和下跌
            data[f'avg_gain_{steps}'] = data.groupby('trade_code')['gain'].transform(
                lambda x: x.rolling(window=steps, min_periods=steps).mean()
            )
            data[f'avg_loss_{steps}'] = data.groupby('trade_code')['loss'].transform(
                lambda x: x.rolling(window=steps, min_periods=steps).mean()
            )

            # 计算 RS
            data[indic_name] = data[f'avg_gain_{steps}'] / data[f'avg_loss_{steps}']

            # 删除中间变量
            data.drop([f'avg_gain_{steps}', f'avg_loss_{steps}'], axis=1, inplace=True)

        # 删除临时列
        data.drop(['delta', 'gain', 'loss'], axis=1, inplace=True)

    def get_vwap(self, data: pd.DataFrame):
        print('Processing VWAP')
        temp = data.loc[data['volume'] != 0]
        temp.reset_index(drop=True, inplace=True)
        temp['vwap'] = temp['amount'] / temp['volume']
        data = data.merge(temp[['trade_code', 'ts', 'vwap']], on=['trade_code', 'ts'], how='left')
        data['vwap_chg'] = data.groupby('trade_code')['vwap'].ffill().pct_change().reset_index(drop=True)
        return data

    def get_corr_optimized(self, data: pd.DataFrame):
        from functools import reduce

        print('Processing Correlation (Optimized)')

        # 确保存在收益率列
        if 'chg' not in data.columns:
            data['chg'] = data.groupby('trade_code')['close'].pct_change()

        # 创建收益率的透视表
        temp = data[['ts', 'trade_code', 'chg']].dropna()
        chg_pivot = temp.pivot(index='ts', columns='trade_code', values='chg').fillna(0)
        chg_pivot.columns = [c.split(self.pair_delimiter)[0] for c in chg_pivot.columns]

        # 获取所有交易代码的列表
        trade_codes = chg_pivot.columns.tolist()

        # 获取所有唯一的交易代码对组合
        from itertools import combinations
        code_pairs = list(combinations(trade_codes, 2))

        # 初始化结果列表
        results = []

        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('corr', rolling_window)
            print(f'Processing rolling window: {rolling_window}')

            # 对收益率数据进行滚动窗口计算
            rolling_corr = chg_pivot.rolling(window=steps).corr()

            # 将相关系数矩阵展开为长格式
            rolling_corr = rolling_corr.stack()
            rolling_corr.name = indic_name
            rolling_corr.index.names = ['ts', 'code1', 'code2']
            rolling_corr = rolling_corr.reset_index()

            # 只保留需要的交易代码对
            rolling_corr = rolling_corr[rolling_corr['code1'] < rolling_corr['code2']]
            rolling_corr['trade_code'] = rolling_corr['code1'] + self.pair_delimiter + rolling_corr['code2']

            # 添加到结果列表
            results.append(rolling_corr[['ts', 'trade_code', indic_name]])

        # 将所有滚动窗口的结果合并
        new_data = reduce(lambda x, y: pd.merge(x, y, on=['ts', 'trade_code'], how='left'), results)
        return new_data

    def get_volume(self, data: pd.DataFrame):
        print('Processing Volume (Optimized)')

        # 过滤成交量不为零的数据
        data_nonzero = data.loc[data['volume'] != 0, ['trade_code', 'ts', 'volume']].copy()

        # 计算成交量的对数值
        data_nonzero['volume_log'] = np.log(data_nonzero['volume'])

        # 按照 trade_code 和 ts 排序，为滚动计算做准备
        data_nonzero.sort_values(['trade_code', 'ts'], inplace=True)

        # 初始化需要合并的列
        merge_columns = ['trade_code', 'ts']

        # 遍历滚动窗口列表
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('volume_log', rolling_window)

            # 对每个 trade_code 计算滚动平均值
            data_nonzero[indic_name] = data_nonzero.groupby('trade_code')['volume_log'].rolling(
                window=steps
            ).mean().reset_index(level=0, drop=True)

            # 计算滚动平均值的百分比变化
            data_nonzero[f'{indic_name}_chg'] = data_nonzero.groupby('trade_code')[indic_name].ffill().pct_change()

            # 添加新列名到合并列表
            merge_columns.extend([indic_name, f'{indic_name}_chg'])

        # 提取需要合并的列
        data_nonzero_merge = data_nonzero[merge_columns]

        # 将计算结果合并回原始数据
        data = data.merge(data_nonzero_merge, on=['trade_code', 'ts'], how='left')

        return data

    def get_vol_pct(self, data: pd.DataFrame):
        print('Processing Vol PCT (Optimized)')

        # 过滤成交量不为0的数据
        data_nonzero = data.loc[data['volume'] != 0, ['trade_code', 'ts', 'volume']].copy()

        # 初始化需要合并的列
        merge_columns = ['trade_code', 'ts']

        # 准备需要的列
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('vol', rolling_window)

            # 计算滚动平均成交量
            data_nonzero[indic_name] = data_nonzero.groupby('trade_code')['volume'].transform(
                lambda x: x.rolling(window=steps).mean()
            ).fillna(0)

            # 计算截面成交量占比
            data_nonzero[f'{indic_name}_pct'] = data_nonzero.groupby('ts')[indic_name].transform(
                lambda x: x / x.sum()
            ).fillna(0)

            # 添加新列名到合并列表
            merge_columns.extend([indic_name, f'{indic_name}_pct'])

        # 提取需要合并的列
        data_nonzero_merge = data_nonzero[merge_columns]

        # 将计算结果合并回原始数据
        data = data.merge(data_nonzero_merge, on=['trade_code', 'ts'], how='left')

        return data

    def get_amt_pct(self, data: pd.DataFrame):
        print('Processing AMT PCT (Optimized)')

        # 过滤成交量不为0的数据
        data_nonzero = data.loc[data['amount'] != 0, ['trade_code', 'ts', 'amount']].copy()

        # 初始化需要合并的列
        merge_columns = ['trade_code', 'ts']

        # 准备需要的列
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('amt', rolling_window)

            # 计算滚动平均成交量
            data_nonzero[indic_name] = data_nonzero.groupby('trade_code')['amount'].transform(
                lambda x: x.rolling(window=steps).mean()
            ).fillna(0)

            # 计算截面成交量占比
            data_nonzero[f'{indic_name}_pct'] = data_nonzero.groupby('ts')[indic_name].transform(
                lambda x: x / x.sum()
            ).fillna(0)

            # 添加新列名到合并列表
            merge_columns.extend([indic_name, f'{indic_name}_pct'])

        # 提取需要合并的列
        data_nonzero_merge = data_nonzero[merge_columns]

        # 将计算结果合并回原始数据
        data = data.merge(data_nonzero_merge, on=['trade_code', 'ts'], how='left')

        return data

    @staticmethod
    def calc_mdd(df):
        peak = df['high'].cummax()
        drawdown = (df['low'] - peak) / peak
        return drawdown.min()

    @staticmethod
    def _calc_group_func(g, steps: int, callback: Callable):
        init_i = g.index.min()
        return g.apply(lambda r: np.nan if r.name - int(steps) < init_i else callback(g.iloc[r.name - steps:r.name, :]),
                       axis=1)

    def get_mdd(self, data: pd.DataFrame):
        print('Processing MDD')
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('mdd', rolling_window)
            # data[indic_name] = data.groupby('trade_code').apply(lambda g: self._calc_group_func(g, steps, self.calc_mdd)).reset_index(drop=True)
            data[indic_name] = data.groupby('trade_code', group_keys=False).apply(
                lambda x: (
                        (x['low'].rolling(window=steps).min() - x['high'].rolling(window=steps).max()) /
                        x['high'].rolling(window=steps).max()
                )
            )

    def get_mru(self, data: pd.DataFrame):
        print('Processing MRU')
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('mru', rolling_window)
            # data[indic_name] = data.groupby('trade_code').apply(lambda g: self._calc_group_func(g, steps, self.calc_mru)).reset_index(drop=True)
            data[indic_name] = data.groupby('trade_code', group_keys=False).apply(
                lambda x: (x['high'].rolling(window=steps).max() - x['open'].shift(steps)) / x['open'].shift(steps))


class CMFeatureGenerator(FeatureGenerator):
    """
        raw_data
        1. 将原始量价数据转换为features，
        2. features存储到本地
        """
    rolling_windows: List[int] = [5, 10, 20, 60, 120]

    def __init__(self, raw_data: pd.DataFrame, bar: str = '1d', period_unit: str = 'D',
                 rolling_windows: List[int] = None, pair_delimiter: str = '.', path: str = None):
        """

        :param raw_data: 原始数据，需包含`ts` `trade_code` `close` `high` `low` `open` `volume`列
        :param bar: 时间周期
        :param path: features最终存储的位置
        """
        super().__init__(raw_data, bar, period_unit, rolling_windows, pair_delimiter, path)

    def __call__(self, save: bool = True, save_method: str = 'replace', start_time: str = None):
        data = self.raw_data.copy().sort_values(by=['ts'])
        data['ts'] = pd.to_datetime(data['ts'])
        if start_time is not None:
            data = data[data.ts >= pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S') - pd.to_timedelta(
                max(self.rolling_windows), unit=self.period_unit)]
        full_data = pd.MultiIndex.from_product([data.trade_code.drop_duplicates().sort_values(),
                                                data.ts.drop_duplicates().sort_values()]).to_frame().reset_index(
            drop=True)
        data = full_data.merge(data, how='left', on=['trade_code', 'ts']).sort_values(by=['trade_code', 'ts'])
        data.reset_index(drop=True, inplace=True)
        data.sort_values(['trade_code', 'ts'], inplace=True)
        data.loc[:, ['pre_settle', 'close', 'open', 'high', 'low', 'settle']] = data.groupby('trade_code')[
            'pre_settle', 'close', 'open', 'high', 'low', 'settle'].ffill().reset_index(drop=True)
        data.loc[:, ['volume', 'oi']] = data[['volume', 'oi']].fillna(0)

        self.get_price_chg(data)
        self.get_mru(data)
        self.get_mdd(data)
        self.get_ma(data)
        self.get_ema(data)
        self.get_volatility(data)
        self.get_macd(data)
        self.get_rsi(data)
        self.get_turnover(data)
        data = self.get_vwap(data)
        data = self.get_volume(data)
        data = self.get_vol_pct(data)
        # data = self.get_amt_pct(data)

        for c in data.columns:
            if c not in ['trade_code', 'ts']:
                temp = data[['trade_code', 'ts', c]].dropna()
                pivot_df = pd.pivot(temp, index='ts', columns=['trade_code'], values=[c])
                pivot_df.columns = [v[1] for v in pivot_df.columns]
                self.features[c] = pivot_df

        corr_data = self.get_corr_optimized(data)
        for c in corr_data.columns:
            if c not in ['trade_code', 'ts']:
                pivot_df = pd.pivot(corr_data, index='ts', columns=['trade_code'], values=[c])
                pivot_df.columns = [v[1] for v in pivot_df.columns]
                self.features[c] = pivot_df

        if save:
            for n, df in self.features.items():
                file_path = f'{self.path}/{n}.pkl'
                if start_time is not None:
                    df = df[df.index >= start_time]

                if save_method == 'append':
                    prev_data = pd.read_pickle(file_path)
                    prev_data = prev_data.loc[prev_data.index < pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S'),
                                :]
                    df = pd.concat([prev_data, df])
                elif save_method == 'replace':
                    pass
                else:
                    Warning('Save method {} not implemented'.format(save_method))
                df.to_pickle(file_path)
        return data, corr_data

    def get_price_chg(self, data: pd.DataFrame):
        super().get_price_chg(data)
        data['settle_close'] = data['close'] / data['settle'] - 1
        data['settle_chg'] = data.groupby('trade_code')['settle'].pct_change()
        data['over_period_settle_chg'] = data['open'] / data['pre_settle'] - 1
        data['settle_high_chg'] = data['high'] / data['pre_settle'] - 1
        data['settle_low_chg'] = data['low'] / data['pre_settle'] - 1
        data['high_settle_chg'] = data['settle'] / data['high'] - 1
        data['low_settle_chg'] = data['settle'] / data['low'] - 1

    def get_turnover(self, data: pd.DataFrame):
        print('Processing Turnover')
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('turnover', rolling_window)
            data[f"{indic_name}_volume"] = data.groupby('trade_code')['volume'].rolling(int(steps)).sum().reset_index(
                drop=True)
            data[f"{indic_name}_oi"] = data.groupby('trade_code')['oi'].rolling(int(steps)).sum().reset_index(drop=True)
            data[indic_name] = data[f"{indic_name}_volume"] / data[f"{indic_name}_oi"]

    def get_macd(self, data: pd.DataFrame):
        print('Processing MACD')
        settings = {'macd_1': [12, 26, 9], 'macd_2': [8, 17, 9]}
        for name, setting in settings.items():
            data['ema1'] = data.groupby('trade_code')['close'].ewm(span=setting[0], adjust=False).mean().reset_index(
                drop=True)
            data['ema2'] = data.groupby('trade_code')['close'].ewm(span=setting[1], adjust=False).mean().reset_index(
                drop=True)
            data['temp'] = data['ema1'] - data['ema2']
            data[name] = data.groupby('trade_code')['temp'].ewm(span=setting[2], adjust=False).mean().reset_index(
                drop=True)
        data.drop(columns=['ema1', 'ema2', 'temp'], inplace=True)


class AShareFeatureGenerator(FeatureGenerator):
    rolling_windows: List[int] = [5, 10, 20, 60, 120]

    def __init__(self, raw_data: pd.DataFrame, bar: str = '1D', period_unit: str = 'D',
                 rolling_windows: List[int] = None, pair_delimiter: str = '.', path: str = None):
        super().__init__(raw_data, bar, period_unit, rolling_windows, pair_delimiter, path)

    def __call__(self, save: bool = True, save_method: str = 'replace', start_time: str = None):
        data = self.raw_data.copy().sort_values(by=['ts'])
        data['ts'] = pd.to_datetime(data['ts'])
        if start_time is not None:
            data = data[data.ts >= pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S') - pd.to_timedelta(
                max(self.rolling_windows), unit=self.period_unit)]
        full_data = pd.MultiIndex.from_product([data.trade_code.drop_duplicates().sort_values(),
                                                data.ts.drop_duplicates().sort_values()]).to_frame().reset_index(
            drop=True)
        data = full_data.merge(data, how='left', on=['trade_code', 'ts']).sort_values(by=['trade_code', 'ts'])
        data.reset_index(drop=True, inplace=True)
        data.sort_values(['trade_code', 'ts'], inplace=True)
        data.loc[:, ['close', 'open', 'high', 'low', 'adjfactor', 'total_shares', 'free_float_shares']] = \
        data.groupby('trade_code')[
            'close', 'open', 'high', 'low', 'adjfactor', 'total_shares', 'free_float_shares'].ffill().reset_index(
            drop=True)
        data.loc[:, ['volume', 'amount']] = data[['volume', 'amount']].fillna(0)

        self.get_price_chg(data)
        self.get_mru(data)
        self.get_mdd(data)
        self.get_ma(data)
        self.get_ema(data)
        self.get_volatility(data)
        self.get_macd(data)
        self.get_rsi(data)
        self.get_free_turnover(data)
        self.get_cap_pct(data)
        data = self.get_vwap(data)
        data = self.get_volume(data)

        for c in data.columns:
            if c not in ['trade_code', 'ts']:
                temp = data[['trade_code', 'ts', c]]
                pivot_df = pd.pivot(temp, index='ts', columns=['trade_code'], values=[c])
                pivot_df.columns = [v[1] for v in pivot_df.columns]
                # pivot_df = pivot_df.fillna(0)
                self.features[c] = pivot_df

        corr_data = self.get_corr_optimized(data)
        for c in corr_data.columns:
            if c not in ['trade_code', 'ts']:
                pivot_df = pd.pivot(corr_data, index='ts', columns=['trade_code'], values=[c])
                pivot_df.columns = [v[1] for v in pivot_df.columns]
                self.features[c] = pivot_df

        if save:
            for n, df in self.features.items():
                file_path = f'{self.path}/{n}.pkl'
                if start_time is not None:
                    df = df[df.index >= start_time]

                if save_method == 'append':
                    prev_data = pd.read_pickle(file_path)
                    prev_data = prev_data.loc[prev_data.index < pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S'),
                                :]
                    df = pd.concat([prev_data, df])
                elif save_method == 'replace':
                    pass
                else:
                    Warning('Save method {} not implemented'.format(save_method))
                df.to_pickle(file_path)
        return data, corr_data

    def get_vwap(self, data: pd.DataFrame):
        print('Processing VWAP')
        temp = data.loc[data['volume'] != 0]
        temp.reset_index(drop=True, inplace=True)
        temp['vwap'] = temp['amount'] / temp['volume']
        data = data.merge(temp[['trade_code', 'ts', 'vwap']], on=['trade_code', 'ts'], how='left')
        data['vwap_chg'] = data.groupby('trade_code')['vwap'].ffill().pct_change().reset_index(drop=True)
        return data

    def get_free_turnover(self, data: pd.DataFrame):
        print('Processing Free Turnover')

        data_nonzero = data.loc[data['volume'] != 0, ['trade_code', 'ts', 'volume', 'free_float_shares']].copy()
        data_nonzero.sort_values(['trade_code', 'ts'], inplace=True)
        merge_columns = ['trade_code', 'ts']

        # 遍历滚动窗口列表
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('free_turnover', rolling_window)
            data_nonzero['vol'] = data_nonzero.groupby('trade_code')['volume'].rolling(
                window=steps
            ).sum().reset_index(level=0, drop=True)
            data_nonzero['share'] = data_nonzero.groupby('trade_code')['free_float_shares'].rolling(
                window=steps
            ).sum().reset_index(level=0, drop=True)

            # 计算滚动平均值的百分比变化
            data_nonzero[indic_name] = data_nonzero['vol'] / data_nonzero['share']

            # 添加新列名到合并列表
            merge_columns.extend([indic_name])

        # 提取需要合并的列
        data_nonzero_merge = data_nonzero[merge_columns]

        # 将计算结果合并回原始数据
        data = data.merge(data_nonzero_merge, on=['trade_code', 'ts'], how='left')

        return data

    def get_cap_pct(self, data: pd.DataFrame):
        print('Processing CAP PCT')

        # 过滤成交量不为0的数据
        data_nonzero = data.loc[data['mkt_cap_ard'] != 0, ['trade_code', 'ts', 'mkt_cap_ard']].copy()

        # 初始化需要合并的列
        merge_columns = ['trade_code', 'ts']

        # 准备需要的列
        for rolling_window in self.rolling_windows:
            steps, indic_name = self._rolling_prepare('cap', rolling_window)

            # 计算滚动平均成交量
            data_nonzero[indic_name] = data_nonzero.groupby('trade_code')['mkt_cap_ard'].transform(
                lambda x: x.rolling(window=steps).mean()
            ).fillna(0)

            # 计算截面成交量占比
            data_nonzero[f'{indic_name}_pct'] = data_nonzero.groupby('ts')[indic_name].transform(
                lambda x: x / x.sum()
            ).fillna(0)

            # 添加新列名到合并列表
            merge_columns.extend([indic_name, f'{indic_name}_pct'])

        # 提取需要合并的列
        data_nonzero_merge = data_nonzero[merge_columns]

        # 将计算结果合并回原始数据
        data = data.merge(data_nonzero_merge, on=['trade_code', 'ts'], how='left')

        return data


class BNFeatureGenerator(FeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, bar: str = '15m',
                 rolling_windows: List[int] = None, pair_delimiter: str = '-', path: str = None):
        super().__init__(raw_data, bar, rolling_windows, pair_delimiter, path)

    def __call__(self, save: bool = True, save_method: str = 'replace',
                 aggregate_len: int = None, start_time: str = None, end_time: str = None, start_hour: int = 1):
        data = self.raw_data.copy().sort_values(by=['ts'])
        data['ts'] = pd.to_datetime(data['ts'])
        data = data.drop(columns=['bar'])
        if start_time is not None:
            data = data[data.ts >= pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S')]
        if end_time is not None:
            data = data[data.ts >= pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')]
        data = self._aggregate_raw_data(data, aggregate_len, start_hour)
        full_data = pd.MultiIndex.from_product([data.trade_code.drop_duplicates().sort_values(),
                                                data.ts.drop_duplicates().sort_values()]).to_frame().reset_index(
            drop=True)
        data = full_data.merge(data, how='left', on=['trade_code', 'ts']).sort_values(by=['trade_code', 'ts'])
        data.reset_index(drop=True, inplace=True)
        data.sort_values(['trade_code', 'ts'], inplace=True)
        data.loc[:, ['close', 'open', 'high', 'low']] = data.groupby('trade_code')[[
            'close', 'open', 'high', 'low']].ffill().reset_index(drop=True)
        data.loc[:, ['volume', 'amount', 'num_trades', 'taker_buy_volume', 'taker_buy_amount']] = data[[
            'volume', 'amount', 'num_trades', 'taker_buy_volume', 'taker_buy_amount']].fillna(0)

        self.get_price_chg(data)
        data = self.get_vwap(data)

        for c in data.columns:
            if c not in ['trade_code', 'ts']:
                temp = data[['trade_code', 'ts', c]].dropna()
                # temp = data[['trade_code', 'ts', c]]
                pivot_df = pd.pivot(temp, index='ts', columns=['trade_code'], values=[c])
                pivot_df.columns = [v[1] for v in pivot_df.columns]
                # pivot_df = pivot_df.fillna(0)
                self.features[c] = pivot_df

        if save:
            for n, df in self.features.items():
                file_path = f'{self.path}/{n}.pkl'
                if start_time is not None:
                    df = df[df.index >= start_time]

                if save_method == 'append':
                    prev_data = pd.read_pickle(file_path)
                    prev_data = prev_data.loc[prev_data.index < pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S'),
                                :]
                    df = pd.concat([prev_data, df])
                elif save_method == 'replace':
                    pass
                else:
                    Warning('Save method {} not implemented'.format(save_method))
                df.to_pickle(file_path)
        return data

    def _aggregate_raw_data(self, data: pd.DataFrame, aggregate_len: int = None, start_hour: int = 1):
        copy_data = data.copy()
        if aggregate_len is not None:
            copy_data.set_index('ts', inplace=True)
            agg_methods = {
                'open': lambda s: s.iloc[0],  # open取聚合后第一条数据
                'high': 'max',  # high取聚合后最高值
                'low': 'min',  # low取聚合后最低值
                'close': lambda s: s.iloc[-1],  # close取聚合后最后一条数据
                'volume': 'sum',  # volume取总和
                'amount': 'sum',  # volume取总和
                'num_trades': 'sum',  # volume取总和
                'taker_buy_volume': 'sum',  # volume取总和
                'taker_buy_amount': 'sum',  # volume取总和
            }
            w = pd.Timedelta(self.bar).total_seconds() * aggregate_len
            # 进行时间聚合，按1小时为窗口进行聚合
            copy_data = copy_data.groupby('trade_code').rolling(aggregate_len).agg(agg_methods)
            copy_data.reset_index(inplace=True)
            copy_data['ts'] = copy_data['ts']
        return copy_data

    def get_price_chg(self, data: pd.DataFrame):
        data['chg'] = data.groupby('trade_code')['close'].pct_change()
        data['over_period_chg'] = data['open'] / data.groupby('trade_code')['close'].shift(1).reset_index(drop=True) - 1
        data['open_chg'] = data.groupby('trade_code')['open'].pct_change().reset_index(drop=True)
        data['high_chg'] = data.groupby('trade_code')['high'].pct_change().reset_index(drop=True)
        data['low_chg'] = data.groupby('trade_code')['low'].pct_change().reset_index(drop=True)
        data['high_begin_chg'] = data['high'] / data['open'] - 1
        data['low_begin_chg'] = data['low'] / data['open'] - 1
        data['high_end_chg'] = data['close'] / data['high'] - 1
        data['low_end_chg'] = data['close'] / data['low'] - 1
        data['amplitude'] = data['high'] / data['low'] - 1
        data['upper_shadow_ratio'] = data['high'] / data[['open', 'close']].max(axis=1) - 1
        data['lower_shadow_ratio'] = data[['open', 'close']].min(axis=1) / data['low'] - 1
        data['body_length_ratio'] = (data['open'] - data['close']).abs() / (data['high'] - data['low'])
        data['body_length_ratio'] = data['body_length_ratio'].fillna(0)
        data['amount_log'] = np.log(data['amount'])
        data['amount_log_chg'] = data.groupby('trade_code')['amount_log'].pct_change()
        data['taker_buy_amount_log'] = np.log(data['taker_buy_amount'])
        data['taker_buy_amount_log_chg'] = data.groupby('trade_code')['taker_buy_amount_log'].pct_change()
        data['num_trades_log'] = np.log(data['num_trades'])
        data['num_trades_log_chg'] = data.groupby('trade_code')['num_trades_log'].pct_change()
        data.drop(['amount_log', 'taker_buy_amount_log', 'num_trades_log'], axis=1, inplace=True)

    def get_vwap(self, data: pd.DataFrame):
        print('Processing VWAP')
        temp = data.loc[data['volume'] != 0]
        temp.reset_index(drop=True, inplace=True)
        temp['vwap'] = temp['amount'] / temp['volume']
        data = data.merge(temp[['trade_code', 'ts', 'vwap']], on=['trade_code', 'ts'], how='left')
        data['vwap_chg'] = data.groupby('trade_code')['vwap'].ffill().pct_change().reset_index(drop=True)

        temp = data.loc[data['taker_buy_volume'] != 0]
        temp.reset_index(drop=True, inplace=True)
        temp['taker_buy_vwap'] = temp['taker_buy_amount'] / temp['taker_buy_volume']
        data = data.merge(temp[['trade_code', 'ts', 'taker_buy_vwap']], on=['trade_code', 'ts'], how='left')
        data['taker_buy_vwap_chg'] = data.groupby('trade_code')['taker_buy_vwap'].ffill().pct_change().reset_index(
            drop=True)
        return data
