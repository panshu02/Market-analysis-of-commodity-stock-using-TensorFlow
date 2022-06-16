import pandas as pd
import numpy as np


def build(data, n_time_lags):

    rename_dic = {'Date': 'Date', 'Vol.': 'Vol', 'Price': 'Price'}
    df = pd.DataFrame(data, columns=['Date', 'Vol.', 'Price'])
    df = df.rename(columns = rename_dic)
    
    k = len(df.index)

    X_cols = []
    for i in range(len(df.index)):
        if df.loc[i, 'Vol'] != '-':
            df.loc[i, 'Vol'] = float(df.loc[i, 'Vol'][:-1])*1000
        try:
            df.loc[i, 'Price'] = float(df.loc[i, 'Price'])
        except:
            price = df.loc[i, 'Price']
            price_li = price.split(',')
            price = ''
            for j in price_li:
                price += j
            df.loc[i, 'Price'] = float(price)

    for i in range(n_time_lags):

        vol_col = df.loc[i+1:, 'Vol']
        vol_col.index = [j for j in range(k-(i+1))]
        prc_col = df.loc[i+1:, 'Price']
        prc_col.index = [j for j in range(k-(i+1))]

        df['Vol({})'.format(i+1)] = vol_col
        df['Price({})'.format(i+1)] = prc_col
        X_cols.append('Vol({})'.format(i+1))
        X_cols.append('Price({})'.format(i+1))
    
    df = df.iloc[:k-n_time_lags, :]
    n = len(df.columns)
    
    for i in range(k-n_time_lags):
        for j in range(n):
            if df.iloc[i, j] == '-':
                df.iloc[i, j] = np.nan
    
    for j in range(1, n):
        df.iloc[:, j] = pd.to_numeric(df.iloc[:, j])

    df = df.interpolate(method = 'polynomial' , order = 2)
    df = df.interpolate(method = 'linear', limit_direction='backward')

    return df, X_cols
