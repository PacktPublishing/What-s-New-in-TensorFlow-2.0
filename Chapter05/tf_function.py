import tensorflow as tf


def custom_function(x, y):
    _sum = tf.add(x, y)
    _diff = tf.subtract(x, y)

    return tf.add(_sum, _diff)


custom_function_graph = tf.function(custom_function)

a = tf.constant(11)
b = tf.constant(1)

print(type(custom_function))
print(type(custom_function_graph))
print(custom_function_graph(a, b))

# Using decorators
@tf.function
def custom_function(x, y):
    _sum = tf.add(x, y)
    _diff = tf.subtract(x, y)

    return tf.add(_sum, _diff)


p = tf.constant(11)
q = tf.constant(1)
print(custom_function(p, q))
