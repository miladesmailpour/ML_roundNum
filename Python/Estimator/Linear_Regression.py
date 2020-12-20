# import tensorflow as tf
#
#
# # y = Wx + b
#
# x_train = [1.0, 2.0, 3.0, 4.0]
# y_train = [-1.0, -2.0, -3.0, -4.0]
#
# w = tf.compat.v1.Variable(initial_value=[1.0], dtype=tf.float32)
# b = tf.compat.v1.Variable(initial_value=[1.0], dtype=tf.float32)
#
# x = tf.compat.v1.placeholder(dtype=tf.float32)
# y_input = tf.compat.v1.placeholder(dtype=tf.float32)
#
# y_output = w * + b
#
# loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train_step = optimizer.minimize(loss=loss)
#
# session = tf.compat.v1.Session()
# session.run(tf.compat.v1.global_variables_initializer)
