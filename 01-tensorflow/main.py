import tensorflow as tf

# This is the simplest way to make tensors
# They have no degrees/ranks (they are scalar)
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(234, tf.int16)
floating = tf.Variable(3.435, tf.float64)

# These tensors have ranks (lists and nested lists)
rank1_tensor = tf.Variable(["Test", "ok", "john"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

# used to determine the rank of tensors
print(tf.rank(rank1_tensor))


# shapes of tensors
print(rank1_tensor.shape)

# changing the shape of tensors (flattening tensors)
tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.reshape(tensor2, [3, -1])

print(tensor1)
print(tensor2)
print(tensor3)

# types of tensors
# Mainly just variables (values that will change) and constants (values that dont change)


t = tf.zeros([5,5,5,5])
# print(t)
t = tf.reshape(t, [125, -1])
print(t)