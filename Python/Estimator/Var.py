import tensorflow as tf

tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()

var_1 = tf.Variable(initial_value=[1.0],
                    trainable=True,
                    validate_shape=True,
                    caching_device=None,
                    name='var_1',
                    variable_def=None,
                    dtype=tf.float32,
                    import_scope=None)
init = tf.compat.v1.global_variables_initializer()


def var_declaration(txt):
    print(f'>_ {txt}')
    print(f'Before initialize \n {var_1}')
    session.run(init)
    print(f'After initialize \n {session.run(fetches=var_1)}')


def var_assign(txt):
    print(f'>_ {txt}')
    var_2 = var_1.assign(value=[2.0])
    print(f'var_1 \n {session.run(fetches=var_1)}')
    print(f'var_2 \n {session.run(fetches=var_2)}')