import tensorflow as tf

def train(X, Y):
    from tensorflow.keras.losses import MeanSquaredLogarithmicError
    mlp = tf.keras.models.Sequential()
    mlp.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
    mlp.add(tf.keras.layers.Dense(units = 10, activation = 'tanh'))
    mlp.add(tf.keras.layers.Dense(units = 2, activation = 'relu'))

    msle = MeanSquaredLogarithmicError()
    mlp.compile(optimizer = 'adam', loss=msle, metrics =[msle])

    print("\n**\tStarting training on the given dataset\t**")
    mlp.fit(X, Y, epochs = 40, batch_size = 3)
    print("\n**\tTraining Finished\t**")
    return  mlp