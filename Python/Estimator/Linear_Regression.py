import tensorflow as tf

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()
# y = -x + b
# y = Wx + b

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]


def lr_model(txt):
    w = tf.compat.v1.Variable(initial_value=[1.0], dtype=tf.compat.v1.float32)
    b = tf.compat.v1.Variable(initial_value=[1.0], dtype=tf.compat.v1.float32)

    x = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32)
    y_input = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32)
    y_output = w * x + b

    loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(loss=loss)

    init = tf.compat.v1.global_variables_initializer()
    session.run(init)

    print('loss:', session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train}))

    for _ in range(1000):
        session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})

    print('loss, w, b after train in order: ', session.run(fetches=[loss, w, b], feed_dict={x: x_train, y_input: y_train}))
    print('Testing the model with new input: ', session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
