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


def load_csv_files_from_path(path, ignore_corr: bool = False, codes: List[str] = None, indics: List[str] = None):
    # 获取目录中的所有文件
    files = os.listdir(path)

    # 筛选出所有的 .csv 文件
    csv_files = [f for f in files if f.endswith('.csv')]

    # 初始化一个空字典来存储读取的数据
    data_dict = {}

    # 依次读取每个 .csv 文件并存入字典
    for csv_file in csv_files:
        if ignore_corr and 'corr' in csv_file:
            continue
        if indics is not None and csv_file not in [f"{i}.csv" for i in indics]:
            continue
        file_path = os.path.join(path, csv_file)
        data = pd.read_csv(file_path)  # 读取 .csv 文件
        if codes is not None:
            if 'corr' in csv_file:
                tem_corr_codes = [c.replace('-USDT', '') for c in codes]
                tem_corr_codes = pydash.flatten([[f'{a}-{b}' for b in tem_corr_codes] for a in tem_corr_codes])
                real_corr_codes = data.columns.tolist()
                corr_codes = [c for c in tem_corr_codes if c in real_corr_codes]
                data = data.loc[:, np.intersect1d(data.columns, corr_codes)]
            else:
                data = data.loc[:, np.intersect1d(data.columns, codes)]
        data_dict[csv_file.replace('.csv', '')] = data  # 将数据存入字典，以文件名为键

    return data_dict


def load_pkl_files_from_path(path, ignore_corr: bool = False, codes: List[str] = None, indics: List[str] = None):
    # 获取目录中的所有文件
    files = os.listdir(path)

    # 筛选出所有的 .pkl 文件
    pkl_files = [f for f in files if f.endswith('.pkl')]

    # 初始化一个空字典来存储读取的数据
    data_dict = {}

    # 依次读取每个 .pkl 文件并存入字典
    for pkl_file in pkl_files:
        if ignore_corr and 'corr' in pkl_file:
            continue
        if indics is not None and pkl_file not in [f"{i}.pkl" for i in indics]:
            continue
        file_path = os.path.join(path, pkl_file)
        data = pd.read_pickle(file_path)  # 读取 .pkl 文件
        if codes is not None:
            if 'corr' in pkl_file:
                tem_corr_codes = [c.replace('-USDT', '') for c in codes]
                tem_corr_codes = pydash.flatten([[f'{a}-{b}' for b in tem_corr_codes] for a in tem_corr_codes])
                real_corr_codes = data.columns.tolist()
                corr_codes = [c for c in tem_corr_codes if c in real_corr_codes]
                data = data.loc[:, np.intersect1d(data.columns, corr_codes)]
            else:
                data = data.loc[:, np.intersect1d(data.columns, codes)]
        data_dict[pkl_file.replace('.pkl', '')] = data  # 将数据存入字典，以文件名为键

    return data_dict

class FeatureHandler:
    """
    1. 提取features
    """
    path = FEATURE_DEFAULT_PATH
    raw_features: Dict[str, pd.DataFrame]
    features: pd.DataFrame

    def __init__(self, pair_delimiter: str = '-', path: str = None, ignore_corr: bool = False, **kwargs):
        self.pair_delimiter = pair_delimiter

        if path is not None:
            self.path = path

        self.codes = kwargs.get('codes')
        self.indics = kwargs.get('indics')

        self.raw_features = load_pkl_files_from_path(self.path, ignore_corr, codes=self.codes, indics=self.indics)
        self.aft_process_features()
        self.features = pd.DataFrame()

    def combine(self, indics: List[str] = None, codes: List[str] = None, begin: str = None, end: str = None,
                dropna: bool = False, scalable: bool = True):
        if indics is None:
            indics = self.indics
        if codes is None:
            codes = self.codes

        if indics is None:
            indics = slice(None)
        if codes is None:
            codes = slice(None)

        if scalable:
            scaled_features = {k: self.scaler(v) for k, v in self.raw_features.items() if k in indics}
        else:
            scaled_features = self.raw_features

        # non corr features
        non_corr_features_combined = pd.concat(scaled_features, axis=1, keys=scaled_features.keys()).loc[:, (indics, codes)]
        # corr features
        corr_features_combined = pd.DataFrame()
        corr_indics = [i for i in indics if 'corr' in i]
        if corr_indics:
            tem_corr_codes = [c.replace('-USDT', '') for c in codes]
            tem_corr_codes = pydash.flatten([[f'{a}-{b}' for b in tem_corr_codes] for a in tem_corr_codes])
            real_corr_codes = scaled_features[corr_indics[0]].columns.tolist()
            corr_codes = [c for c in tem_corr_codes if c in real_corr_codes]
            corr_dict = pydash.pick(scaled_features, corr_indics)
            corr_features_combined = pd.concat(corr_dict.values(), axis=1, keys=corr_indics).loc[:, (corr_indics, corr_codes)]

        features_combined = pd.concat([non_corr_features_combined, corr_features_combined], axis=1)

        if dropna:
            features_combined.dropna(axis=0, how='any', inplace=True)

        if begin is not None:
            features_combined = features_combined.loc[features_combined.index >= pd.to_datetime(begin), :]
        if end is not None:
            features_combined = features_combined.loc[features_combined.index <= pd.to_datetime(end), :]

        self.features = features_combined
        return self

    def aft_process_features(self):
        for k, v in self.raw_features.items():
            if k in ['close', 'open', 'high', 'low']:
                cash_col = pd.DataFrame(1, index=v.index, columns=['USDT-USDT'])
            else:
                cash_col = pd.DataFrame(0, index=v.index, columns=['USDT-USDT'])
            self.raw_features[k] = pd.concat([v, cash_col], axis=1)
        # for i in self.raw_features.keys():
        #     # vwap_chg
        #     if i == 'vwap_chg':
        #         temp = self.get_raw_feature(i)
        #         if temp is not None:
        #             temp.iloc[0] = np.nan
        #             self.raw_features[i] = temp
        #     # rsi
        #     if 'rsi' in i:
        #         temp = self.get_raw_feature(i)
        #         if temp is not None:
        #             temp = 100 / (100 - temp) - 1
        #             temp[np.isinf(temp)] = 0
        #             self.raw_features[i] = temp
        #     # # vol amt
        #     # if 'amt_pct' in i or 'volume_log_chg' in i or 'vol_pct' in i:
        #     #     temp = self.get_raw_feature(i)
        #     #     if temp is not None:
        #     #         temp[(temp.shift(1) == 0).shift(-1).fillna(False)] = np.NaN
        #     #         self.raw_features[i] = temp
        # pass

    @staticmethod
    def scaler(input):
        scaler = StandardScaler()

        output = scaler.fit_transform(input.values.transpose()).transpose()
        output = pd.DataFrame(output, columns=input.columns, index=input.index)
        return output

    def get_features(self):
        return self.features

    def get_raw_feature(self, indic_name: str):
        return self.raw_features.get(indic_name)
