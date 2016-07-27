import numpy as np
import time
import sys

cur_time = time.time()
mb_dim = 32 #training examples per minibatch
x_dim = 28  #size of one side of square image
y_dim = int(sys.argv[1])  #possible classes
y_dim_alt = int(sys.argv[2])
n_samples_per_class = 1 #samples of each class

eps = 1e-10 #term added for numerical stability of log computations
tie = False #tie the weights of the query network to the labeled network
x_i_learn = True #toggle learning for the query network
learning_rate = 1e-1

data = np.load('data.npy')
data = np.reshape(data,[-1,20,28,28]) #each of the 1600 classes has 20 examples

'''
    Samples a minibatch of size mb_dim. Each training example contains
    n_samples labeled samples, such that n_samples_per_class samples
    come from each of y_dim randomly chosen classes. An additional example
    one one of these classes is then chosen to be the query, and its label
    is the target of the network.
'''
def get_minibatch(num_classes):
    n_samples = num_classes * n_samples_per_class
    mb_x_i = np.zeros((mb_dim,n_samples,x_dim,x_dim,1))
    mb_y_i = np.zeros((mb_dim,n_samples))
    mb_x_hat = np.zeros((mb_dim,x_dim,x_dim,1),dtype=np.int)
    mb_y_hat = np.zeros((mb_dim,),dtype=np.int)
    for i in range(mb_dim):
        ind = 0
        pinds = np.random.permutation(n_samples)
        classes = np.random.choice(data.shape[0],num_classes,False)
        x_hat_class = np.random.randint(num_classes)
        for j,cur_class in enumerate(classes): #each class
            example_inds = np.random.choice(data.shape[1],n_samples_per_class,False)
            for eind in example_inds:
                mb_x_i[i,pinds[ind],:,:,0] = np.rot90(data[cur_class][eind],np.random.randint(4))
                mb_y_i[i,pinds[ind]] = j
                ind +=1
            if j == x_hat_class:
                mb_x_hat[i,:,:,0] = np.rot90(data[cur_class][np.random.choice(data.shape[1])],np.random.randint(4))
                mb_y_hat[i] = j
    return mb_x_i,mb_y_i,mb_x_hat,mb_y_hat



                



import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/oneshot_logs', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)


x_hat = tf.placeholder(tf.float32,shape=[None,x_dim,x_dim,1])
x_i = tf.placeholder(tf.float32,shape=[None,y_dim * n_samples_per_class ,x_dim,x_dim,1])
y_i_ind = tf.placeholder(tf.int64,shape=[None,y_dim * n_samples_per_class])
y_i = tf.cast(tf.one_hot(y_i_ind,y_dim, 1, 0), tf.float32)
y_hat_ind = tf.placeholder(tf.int64,shape=[None])
y_hat = tf.cast(tf.one_hot(y_hat_ind,y_dim, 1, 0), tf.float32)

x_hat_alt = tf.placeholder(tf.float32,shape=[None,x_dim,x_dim,1])
x_i_alt = tf.placeholder(tf.float32,shape=[None, y_dim_alt * n_samples_per_class,x_dim,x_dim,1])
y_i_ind_alt = tf.placeholder(tf.int64,shape=[None,y_dim_alt * n_samples_per_class])
y_i_alt = tf.cast(tf.one_hot(y_i_ind_alt,y_dim_alt, 1, 0), tf.float32)
y_hat_ind_alt = tf.placeholder(tf.int64,shape=[None])
y_hat_alt = tf.cast(tf.one_hot(y_hat_ind_alt,y_dim_alt, 1, 0), tf.float32)
'''
    creates a stack of 4 layers. Each layer contains a
    3x3 conv layers, batch normalization, retified activation,
    and then 2x2 max pooling. The net effect is to tranform the
    mb_dimx28x28x1 images into a mb_dimx1x1x64 embedding, the extra
    dims are removed, resulting in mb_dimx64.
'''
def make_conv_net(inp,scope,reuse=None,stop_grad=False, index=0, alt_reuse=None):
    with tf.variable_scope(scope) as varscope:
        cur_input = inp
        cur_filters = 1
        for i in range(4):
            with tf.variable_scope('conv'+str(i), reuse=reuse):
                W = tf.get_variable('W',[3,3,cur_filters,64])
            #with tf.variable_scope('conv' + str(i) + '_' + str(index), reuse = alt_reuse):
                beta = tf.get_variable('beta', shape=[64], initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable('gamma', shape=[64], initializer=tf.constant_initializer(1.0))
                cur_filters = 64
                pre_norm = tf.nn.conv2d(cur_input,W,strides=[1,1,1,1],padding='SAME')
                mean,variance = tf.nn.moments(pre_norm,[0,1,2])
                post_norm = tf.nn.batch_normalization(pre_norm,mean,variance,beta,gamma,eps)
                conv = tf.nn.relu(post_norm)
                cur_input = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    if stop_grad:
        return tf.stop_gradient(tf.squeeze(cur_input,[1,2]))
    else:
        return tf.squeeze(cur_input,[1,2])
'''
    assemble a computational graph for processing minibatches of the n_samples labeled examples and one unlabeled sample.
    All labeled examples use the same convolutional network, whereas the unlabeled sample defaults to using different parameters.
    After using the convolutional networks to encode the input, the pairwise cos similarity is computed. The normalized version of this
    is used to weight each label's contribution to the queried label prediction.
'''
scope = 'encode_x'
x_hat_encode = make_conv_net(x_hat,scope)
#x_hat_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_hat_encode),1,keep_dims=True),eps,float("inf")))
cos_sim_list = []
if not tie:
    scope = 'encode_x_i'
for i in range(y_dim * n_samples_per_class):
    x_i_encode = make_conv_net(x_i[:,i,:,:,:],scope,tie or i > 0,not x_i_learn, index=i)
    x_i_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_i_encode),1,keep_dims=True),eps,float("inf")))
    dotted = tf.squeeze(
        tf.batch_matmul(tf.expand_dims(x_hat_encode,1),tf.expand_dims(x_i_encode,2)),[1,])
    cos_sim_list.append(dotted
            *x_i_inv_mag)
            #*x_hat_inv_mag
cos_sim = tf.concat(1,cos_sim_list)
tf.histogram_summary('cos sim',cos_sim)
weighting = tf.nn.softmax(cos_sim)
label_prob = tf.squeeze(tf.batch_matmul(tf.expand_dims(weighting,1),y_i))
tf.histogram_summary('label prob',label_prob)

top_k = tf.nn.in_top_k(label_prob,y_hat_ind,1)
acc = tf.reduce_mean(tf.to_float(top_k))
tf.scalar_summary('avg accuracy',acc)
correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(label_prob,eps,1.0))*y_hat,1)
loss = tf.reduce_mean(-correct_prob,0)
tf.scalar_summary('loss',loss)
optim = tf.train.GradientDescentOptimizer(learning_rate)
#optim = tf.train.AdamOptimizer(learning_rate)
grads = optim.compute_gradients(loss)
grad_summaries = [tf.histogram_summary(v.name,g) if g is not None else '' for g,v in grads]
train_step = optim.apply_gradients(grads)

'''
    End of the construction of the computational graph.
'''

''' Construct alternative (i.e. different y dimension) computational graph '''

scope = 'encode_x'
x_hat_encode_alt = make_conv_net(x_hat_alt,scope, reuse=True, alt_reuse = True)
#x_hat_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_hat_encode),1,keep_dims=True),eps,float("inf")))
cos_sim_list_alt = []
if not tie:
    scope = 'encode_x_i'
for i in range(y_dim_alt * n_samples_per_class):
    x_i_encode_alt = make_conv_net(x_i_alt[:,i,:,:,:],scope,True,not x_i_learn, index=i, alt_reuse = True)
    x_i_inv_mag_alt = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_i_encode_alt),1,keep_dims=True),eps,float("inf")))
    dotted_alt = tf.squeeze(tf.batch_matmul(tf.expand_dims(x_hat_encode_alt,1),tf.expand_dims(x_i_encode_alt,2)),[1,])
    cos_sim_list_alt.append(dotted_alt*x_i_inv_mag_alt)
cos_sim_alt = tf.concat(1,cos_sim_list_alt)
tf.histogram_summary('cos sim alt',cos_sim_alt)
weighting_alt = tf.nn.softmax(cos_sim_alt)
label_prob_alt = tf.squeeze(tf.batch_matmul(tf.expand_dims(weighting_alt,1),y_i_alt))
tf.histogram_summary('label prob alt',label_prob_alt)

top_k_alt = tf.nn.in_top_k(label_prob_alt,y_hat_ind_alt,1)
acc_alt = tf.reduce_mean(tf.to_float(top_k_alt))
tf.scalar_summary('avg accuracy alt',acc_alt)
correct_prob_alt = tf.reduce_sum(tf.log(tf.clip_by_value(label_prob_alt,eps,1.0))*y_hat_alt,1)
loss_alt = tf.reduce_mean(-correct_prob_alt,0)
tf.scalar_summary('loss alt',loss_alt)


''' End of the construction of the alternative computational graph '''

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(sys.argv[3], graph=sess.graph)
sess.run(tf.initialize_all_variables())

for v in tf.trainable_variables():
    print(v.name)

for i in range(int(1e7)):
    mb_x_i,mb_y_i,mb_x_hat,mb_y_hat = get_minibatch(y_dim)
    mb_x_i_alt,mb_y_i_alt,mb_x_hat_alt,mb_y_hat_alt = get_minibatch(y_dim_alt)
    feed_dict = {x_hat: mb_x_hat,
                y_hat_ind: mb_y_hat,
                x_i: mb_x_i,
                y_i_ind: mb_y_i,
                x_hat_alt: mb_x_hat_alt,
                y_hat_ind_alt: mb_y_hat_alt,
                x_i_alt: mb_x_i_alt,
                y_i_ind_alt: mb_y_i_alt}

    _,mb_loss, mb_acc, mb_loss_alt, mb_acc_alt, summary,ans = sess.run([train_step,loss, acc, loss_alt, acc_alt, merged,cos_sim],feed_dict=feed_dict)
    if i % int(10) == 0:
        print(i,'loss: ',mb_loss, 'acc: ', mb_acc, 'loss alt: ', mb_loss_alt, 'acc alt: ', mb_acc_alt, 'time: ',time.time()-cur_time)
        cur_time = time.time()
        if i % int(100) == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            writer.add_run_metadata(run_metadata, 'step%d' % i)
    writer.add_summary(summary,i)

sess.close()



