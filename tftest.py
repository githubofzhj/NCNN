import tensorflow as tf

a=tf.Variable(tf.random_normal([2,4]),tf.float32)
b=tf.Variable(tf.random_normal([2,2,3]),tf.float32)
c=tf.Variable(tf.random_normal([2,3]),tf.float32)
e=tf.expand_dims(a,0)
d=tf.concat([e,b],-1)

em=tf.nn.embedding_lookup(a,[[0,1],[0,1]])
p1=tf.nn.embedding_lookup(b,[[0,1],[0,1]])
e=tf.expand_dims(a,0)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('a\n',sess.run(a))
    print('b\n', sess.run(b))

    # print('em\n',sess.run(em))
    # print('p1\n', sess.run(p1))
    print('d\n',sess.run(d))
