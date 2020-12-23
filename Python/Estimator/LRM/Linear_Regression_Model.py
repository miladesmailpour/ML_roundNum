import tensorflow as tf

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()
# y = -x + b
# y = Wx + b

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]


def lr_model(txt):
    w = tf.compat.v1.Variable(initial_value=[1.0], dtype=tf.compat.v1.float32, name='w')
    b = tf.compat.v1.Variable(initial_value=[1.0], dtype=tf.compat.v1.float32, name='b')

    x = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, name='x')
    y_input = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, name='y_input')
    # y_output = w * x + b
    # change line following line to tensorflow node in line below
    y_output = tf.add(x=tf.multiply(x=w, y=x, name='multiply'), y=b, name='t_output')

    loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input), name='loss')
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01, name='optimizer')
    train_step = optimizer.minimize(loss=loss, name='train_step')
    # Save the Graph in order step1
    saver = tf.compat.v1.train.Saver()

    init = tf.compat.v1.global_variables_initializer()
    session.run(init)

    # Write the graph step2
    tf.compat.v1.train.write_graph(graph_or_graph_def=session.graph_def,
                                   logdir='.',
                                   name='LRM/linear_regression.pbtxt',
                                   as_text=False)
    print('loss:', session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train}))

    for _ in range(1000):
        session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})

    # saving the graph step3
    saver.save(sess=session, save_path='LRM/linear_regression.ckpt')

    print('loss, w, b after train in order: ', session.run(fetches=[loss, w, b], feed_dict={x: x_train, y_input: y_train}))
    print('Testing the model with new input: ', session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
