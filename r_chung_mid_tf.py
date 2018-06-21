import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    print(input_tensor)
    b_out, step, summary = sess.run([b, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)
    print(type(b_out), step)

graph = tf.Graph()

with graph.as_default():
    # Creates a placeholder vector of user defined length with data type int32
    with tf.name_scope("Variables"):
#         previous_value = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="previous_value")
#         update_value = tf.assign(b)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        increment_step = global_step.assign_add(1)

    with tf.name_scope("Input_vector"):
        a = tf.placeholder(tf.float32, name="my_vector")

    # calculate SMA moving average
    with tf.name_scope("SMA_section"):
        b = tf.reduce_mean(a, name="SMA")
        tf.summary.scalar('sma_mean', b)

    with tf.name_scope("SMA_summary"):
        init = tf.global_variables_initializer()
        merged_summaries = tf.summary.merge_all()

#save summary
# To run we have to add the two extra lines or run them in the shell:
sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./moving_average', graph)
sess.run(init)

## User input to define the length of your vector and the window width
random_length = int(input("Enter the length of your random vector: "))
window_width = int(input("Enter the length of the window width: "))
print("The step of SMA will be 1\n")
vector = np.random.normal(1, 2, random_length)
print(vector)
print(vector.shape)

i = 0
j = i+window_width
while j <= len(vector):
#     print(i)
    input_vector = vector[i:j]
    i += 1
    j += 1
    print(vector)
#     print(type(input_vector))
    print(input_vector.shape)

    run_graph(input_vector)

writer.flush()
writer.close()
sess.close()
