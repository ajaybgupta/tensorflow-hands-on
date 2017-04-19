import tensorflow as tf

const_scalar_one = tf.constant(1.0)
const_scalar_two = tf.constant(1.0, tf.float32)
const_scalar_three = tf.constant(1.0, tf.float32, name="const_scalar_three")

const_vector_one = tf.constant([2, 2], name="const_vector_one")
const_matrix_one = tf.constant([[2, 2], [1, 1]], name="const_matrix_one")
# We can have n dimension tensor - Useful while computation
const_tensor_3D = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], name="const_tensor_3D")

# Pre Filled Values Tensor
# Pre Filled Zeros
const_vector_two = tf.zeros([2, 3], tf.int32, name="const_vector_two")
const_vector_three = tf.zeros_like(const_vector_two, name="const_vector_three")
# Pre Filled Ones
const_vector_four = tf.ones([2, 3], tf.int32, name="const_vector_four")
const_vector_five = tf.ones_like(const_vector_four, tf.int32, name="const_vector_five")
# Pre Filled Some Specified Number
const_vector_six = tf.fill([2, 3], 8, name="const_vector_six")

# Sequence Constant
# Lin Space is used for reaching from start to end and in step.
# As we have specified as step 4 the output is 10 11 12 13
# If we specify step as 7 it will give output as 10 10.5 11 11.5 till it reaches 13.0
const_vector_seven = tf.linspace(10.0, 13.0, 4, name="const_vector_seven")

with tf.Session() as sess:
    print(sess.run(const_scalar_one))
    print(sess.run(const_scalar_two))
    print(sess.run(const_scalar_three))
    print(sess.run(const_vector_one))
    print(sess.run(const_matrix_one))
    print(sess.run(const_vector_two))
    print(sess.run(const_vector_three))
    print(sess.run(const_vector_four))
    print(sess.run(const_vector_five))
    print(sess.run(const_vector_six))
    print(sess.run(const_vector_seven))
