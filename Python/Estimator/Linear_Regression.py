import tensorflow as tf

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()
##############################
# y = mx + b  OR  y = wx + b #
##############################
w = tf.compat.v1.constant([2.0])
b = tf.compat.v1.constant([1.0])
x = tf.compat.v1.placeholder(dtype=tf.float32)


def lr_not_tf_operation(txt):
    print(f'>_ {txt}')
    y = w * x + b
    print(session.run(fetches=y, feed_dict={x: [2.0]}))


def lr_tf_operation(txt):
    print(f'>_ {txt}')
    multi = tf.multiply(x=w, y=x)
    y = tf.add(x=multi, y=b)
    print('__Single input:')
    print(session.run(fetches=y, feed_dict={x: [2.0]}))
    print('__more than one input:')
    print(session.run(fetches=y, feed_dict={x: [2.0, 3.0, 4.0]}))
