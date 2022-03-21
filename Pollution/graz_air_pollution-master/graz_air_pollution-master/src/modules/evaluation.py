import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime


class calc_errors:
    """

    """

    def __init__(self, df):
        self.df = df
        self.cols_pred = self.df.filter(regex='pred')
        self.cols_true = self.df[~self.df.isin(self.cols_pred)].dropna(axis=1, how='all')
        for col in self.cols_pred.columns:
            self.cols_pred = self.cols_pred.rename(columns={col: col[:-5]})

    def calc_mse(self):
        mse = {}
        for col1, col2 in zip(self.cols_true.columns, self.cols_pred.columns):
            if col1 == col2:
                mse[col1] = mean_squared_error(self.cols_true[col1], self.cols_pred[col2], squared=False)
        return mse

    def calc_mae(self):
        mae = {}
        for col1, col2 in zip(self.cols_true.columns, self.cols_pred.columns):
            if col1 == col2:
                mae[col1] = mean_absolute_error(self.cols_true[col1], self.cols_pred[col2])
        return mae

    def calc_r2(self):
        r2 = {}
        for col1, col2 in zip(self.cols_true.columns, self.cols_pred.columns):
            if col1 == col2:
                r2[col1] = r2_score(self.cols_true[col1], self.cols_pred[col2])
        return r2
