from get.api_get import give_df
from model.build import build
from model.make_trainset import trainset
from model.train import train
from model.predict_results import predict_result
import pandas as pd
import numpy as np

if __name__ == '__main__':
    urls = dict()
    urls['Crude Oil'] = 'https://www.investing.com/commodities/crude-oil-historical-data'
    urls['Natural Gas'] = 'https://www.investing.com/commodities/natural-gas-historical-data'
    urls['Gold'] = 'https://www.investing.com/commodities/gold-historical-data'
    urls['Bitcoin'] = 'https://www.investing.com/crypto/bitcoin/historical-data'
    urls['Ethereum'] = 'https://www.investing.com/crypto/ethereum/historical-data'
    for commodity in urls:
        url = urls[commodity]
        data = give_df(url)
        print("\n**\t{} past trends\t**\n\n".format(commodity))
        data = give_df(url)

        n_time_lags = 4
        df, X_cols = build(data, 4)
        print("Pre-processed training set:\n")
        print(df)
        sc1, sc2, X, Y = trainset(df, X_cols)

        mlp = train(X, Y)

        cols = ['Vol', 'Price']
        for i in range(1, n_time_lags):
            cols.append('Vol({})'.format(i))
            cols.append('Price({})'.format(i))

        next_day = df.loc[0, cols].values
        #print(next_day)
        next_day = sc1.transform([next_day])

        results = pd.DataFrame(columns = df.columns[1:])
        #print(len(results.columns))

        result = mlp.predict(next_day)
        result = sc2.inverse_transform(result)
        print("\n Prediction for price and volume for tommorow:")
        print("Price($) : {}\t\t Volume(Barrels) : {}\n".format(result[0][1], result[0][0]))
        next_day = sc1.inverse_transform(next_day)
        #print(next_day)

        results.loc[0] = np.concatenate((result[0], next_day[0]))
        #print(results)

        n_days = 3
        results = predict_result(n_days, results, cols, mlp, sc1, sc2)
        print("Price and Volume predictions for the next {} days:\n".format(n_days))
        print(results.loc[:, ['Price', 'Vol']])
