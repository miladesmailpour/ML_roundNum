import tensorflow as tf

x_train = [1.0, 2.0, 3.0, 4.0] # input for training
y_train = [2.0, 3.0, 4.0, 5.0] # expected Value
y_actual = [1.5, 2.5, 3.5, 4.5] # actual output


def simple_loss(txt):
    loss = tf.reduce_sum(input_tensor=tf.square(x=y_train - y_actual))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05)
    train_step = optimizer.minimize(loss=loss)
    print(f'>_ {txt}')
