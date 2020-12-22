import tensorflow as tf
import numpy as np

# y = -x + 0
# inputs = [1.0, 2.0, 3.0, 4.0]
# outputs = [-1.0, -2.0, -3.0, -4.0]

x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([-1.0, -2.0, -3.0, -4.0])
x_eval = np.array([5.0, 10.0, 15.0, 20.0])
y_eval = np.array([-5.1, -10.1, -15.1, -20.1])


def estimator(txt):
    # y = Wx + b
    feature_column = tf.compat.v1.feature_column.numeric_column(key='x', shape=[1])
    features_columns = [feature_column]
    estimator = tf.compat.v1.estimator.LinearRegressor(features_columns=features_columns)
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                                            y=y_train,
                                                            batch_size=4,
                                                            num_epochs=None,
                                                            shuffle=True)
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                                                  y=y_train,
                                                                  batch_size=4,
                                                                  num_epochs=1000,
                                                                  shuffle=False)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_eval},
                                                                 y=y_eval,
                                                                 batch_size=4,
                                                                 num_epochs=1000,
                                                                 shuffle=False)
    estimator.train(input_fn=input_fn,
                    steps=1000)
