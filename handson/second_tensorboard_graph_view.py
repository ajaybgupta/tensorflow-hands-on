import tensorflow as tf

"""Generating Session Graph and Using TensorBoard for Visualization"""

# Number Constant
# Name is added to get name on graph, otherwise TensorFlow will give name as const and const_1
const_one = tf.constant(1.0, tf.float32, name="const_one")
const_two = tf.constant(2.0, name="const_two")

# Addition of Constant
# Name is added to get name on graph
addition = tf.add(const_one, const_two, name="add")

with tf.Session() as sess:
    # Write session graph to file
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(addition))

# Session Graph can be viewed on TensorBoard using this command
# tensorboard --logdir="./graphs"
# Go to graph tab for getting the graph connection between node
# Useful while debugging complex data flows

writer.close()
