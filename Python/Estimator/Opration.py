import tensorflow as tf

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()


def not_tf_operation_constant(txt):
    const_1 = tf.compat.v1.constant([1.0])
    const_2 = tf.compat.v1.constant([2.0])
    result = const_1 + const_2
    print(f'>_ {txt}')
    print(session.run(fetches=result))


def tf_operation_constant(txt):
    const_1 = tf.compat.v1.constant([1.0])
    const_2 = tf.compat.v1.constant([2.0])
    result = tf.add(x=const_1, y=const_2, name='result')
    print(f'>_ {txt}')
    print(session.run(fetches=result))


def tf_operation_placeholder(txt):
    placeholder_1 = tf.compat.v1.placeholder(dtype=tf.float32)
    const_1 = tf.compat.v1.constant([2.0])
    result = tf.add(x=const_1, y=placeholder_1, name='result')
    print(f'>_ {txt}')
    print(session.run(fetches=result, feed_dict={placeholder_1: [2.0]}))

