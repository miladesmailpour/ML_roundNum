import tensorflow as tf

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()


def single_placeholder(txt):
    placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 4), name='placeholder')
    print(placeholder)
    print(f'>_ {txt}')
    print(session.run(fetches=placeholder, feed_dict={placeholder: [[1.0, 2.0, 3.0, 4.0]]}))


def multi_placeholder(txt):
    placeholder_1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 4), name='placeholder_1')
    placeholder_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 2), name='placeholder_2')
    print(placeholder_1)
    print(placeholder_2)
    print(f'>_ {txt}')
    print(session.run(fetches=[placeholder_1, placeholder_2],
                      feed_dict={placeholder_1: [[1.0, 2.0, 3.0, 4.0]], placeholder_2: [[1.0, 2.0], [3.0, 4.0]]}))