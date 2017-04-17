import tensorflow as tf

# String Constant
hello = tf.constant('Hello, TensorFlow!')

# Number Constant
const_one = tf.constant(1.0, tf.float32)
const_two = tf.constant(2.0)
# Addition of Constant
addition = tf.add(const_one, const_two)

sess = tf.Session()
print(sess.run(hello))
print(sess.run(addition))
