import numpy as np

def predict_result(n_days, results, cols, mlp, sc1, sc2):
    for i in range(1, n_days):
        next_day = results.loc[i-1, cols].values
        next_day = sc1.transform([next_day])

        result = mlp.predict(next_day)
        result = sc2.inverse_transform(result)
        next_day = sc1.inverse_transform(next_day)

        results.loc[i] = np.concatenate((result[0], next_day[0]))

    return results