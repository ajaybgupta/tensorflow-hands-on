import tensorflow as tf

# API Docs Reference
# https://www.tensorflow.org/api_guides/python/math_ops#Matrix_Math_Functions

# Scalar
constant_scalar_one = tf.constant(1, name="constant_scalar_one")
constant_scalar_two = tf.constant(2, name="constant_scalar_two")
constant_scalar_three = tf.constant(3, name="constant_scalar_two")
# Vector
constant_vector_one = tf.constant([3, 6], name="constant_vector_one")
constant_vector_two = tf.constant([4, 2], name="constant_vector_two")
constant_vector_three = tf.constant([1, 3], name="constant_vector_three")
# Matrix
constant_matrix_one = tf.constant([[1, 2], [3, 4]], name="constant_matrix_one")
constant_matrix_two = tf.constant([[5, 6], [7, 8]], name="constant_matrix_two")
constant_matrix_three = tf.constant([[9, 10], [11, 12]], name="constant_matrix_three")
constant_matrix_four = tf.constant([[1, 2, 3], [4, 5, 6]], name="constant_matrix_four")

# Addition
# To add two tensor - it can be scalar, vector, matrix or n rank tensor
addition_two_scalar = tf.add(constant_scalar_one, constant_scalar_two, name="addition_two_scalar")
addition_two_vector = tf.add(constant_vector_one, constant_vector_two, name="addition_two_vector")
addition_two_matrix = tf.add(constant_matrix_one, constant_matrix_two, name="addition_two_vector")

# To add multiple tensor - it can be scalar, vector, matrix or tensor with n rank
addition_n_scalar = tf.add_n([constant_scalar_one, constant_scalar_two, constant_scalar_three],
                             name="addition_n_scalar")
addition_n_vector = tf.add_n([constant_vector_one, constant_vector_two, constant_vector_three],
                             name="addition_n_vector")
addition_n_matrix = tf.add_n([constant_matrix_one, constant_matrix_two, constant_matrix_three],
                             name="addition_n_matrix")

# Addition between Scalar and Vector
addition_scalar_vector = tf.add(constant_scalar_one, constant_vector_one, name="addition_scalar_vector")
addition_scalar_matrix = tf.add(constant_scalar_one, constant_matrix_one, name="addition_scalar_matrix")
addition_vector_matrix = tf.add(constant_vector_one, constant_matrix_one, name="addition_vector_matrix")
# addition_vector_matrix_dimension_mismatch = tf.add(constant_vector_one, constant_matrix_four,
#                                                   name="addition_vector_matrix_dimension_mismatch")

# Same holds true for Subtraction
# Subtraction, Multiplication and Division
subtraction_two_vector = tf.subtract(constant_vector_one, constant_vector_two, name="subtraction_two_vector")

# Multiplication - Element Wise
multiplication_two_scalar = tf.multiply(constant_scalar_one, constant_scalar_two, name="multiplication_two_scalar")

# Division
division_two_scalar = tf.divide(constant_scalar_one, constant_scalar_two, name="division_two_scalar")

with tf.Session() as sess:
    print("addition_two_scalar")
    print(sess.run(addition_two_scalar))
    print("addition_two_vector")
    print(sess.run(addition_two_vector))
    print("addition_two_matrix")
    print(sess.run(addition_two_matrix))

    print("addition_scalar_vector")
    print(sess.run(addition_scalar_vector))

    print("addition_scalar_matrix")
    print(sess.run(addition_scalar_matrix))

    print("addition_vector_matrix")
    print(sess.run(addition_vector_matrix))

    # print("addition_vector_matrix_dimension_mismatch")
    # print(sess.run(addition_vector_matrix_dimension_mismatch))

    print("addition_n_scalar")
    print(sess.run(addition_n_scalar))
    print("addition_n_vector")
    print(sess.run(addition_n_vector))
    print("addition_n_matrix")
    print(sess.run(addition_n_matrix))

    print("subtraction_two_vector")
    print(sess.run(subtraction_two_vector))

    print("multiplication_two_scalar")
    print(sess.run(multiplication_two_scalar))

    print("division_two_scalar")
    print(sess.run(division_two_scalar))
    print(1/2)
