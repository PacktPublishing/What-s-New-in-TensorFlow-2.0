import tensorflow as tf


@tf.function
def sum_of_cubes(numbers):

    _sum = 0

    for number in numbers:
        _sum += number ** 3

    return _sum


input_values = tf.constant([1, 2, 3, 4, 5])
result = sum_of_cubes(input_values)
print(type(result))
print(result)
