import pandas as pd

def trainset(df, X_cols):
    X = pd.DataFrame(df, columns=X_cols).values
    Y = pd.DataFrame(df[['Vol', 'Price']]).values
    
    #print("X values for Training\n")
    #print(X)
    #print("\nY values for Training\n")
    #print(Y,'\n')

    from sklearn.preprocessing import StandardScaler
    sc1 = StandardScaler()
    sc2 = StandardScaler()
    X = sc1.fit_transform(X)
    Y = sc2.fit_transform(Y)
    return sc1, sc2, X, Y