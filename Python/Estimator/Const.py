import tensorflow as tf


def simple_1(txt):
    print(f'>_ {txt}')
    const_1 = tf.constant(value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=tf.float32,
                          shape=(2, 3),
                          name='const_1')
    print(const_1)


def simple_2(txt):
    print(f'>_ {txt}')
    with tf.compat.v1.Session() as session:
        const_1 = tf.constant(value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                              dtype=tf.float32,
                              shape=(2, 3),
                              name='const_1')
        print(session.run(fetches=const_1))


def simple_3(txt):
    print(f'>_ {txt}')
    tf.compat.v1.disable_eager_execution()
    session = tf.compat.v1.Session()
    const_1 = tf.constant(value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=tf.float32,
                          shape=(2, 3),
                          name='const_1')
    print(session.run(fetches=const_1))


def simple_4(txt):
    print(f'>_ {txt}')
    tf.compat.v1.disable_eager_execution()
    session = tf.compat.v1.Session()
    const_1 = tf.constant(value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=tf.float32,
                          shape=(2, 3),
                          name='const_1')
    const_2 = tf.constant(value=[[1.0, 3.0]],
                          dtype=tf.float32,
                          shape=(1, 2),
                          name='const_2')
    print(session.run(fetches=[const_1, const_2]))