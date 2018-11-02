#!/usr/bin/env python
import tensorflow as tf
import numpy as np 


filename_queue = tf.train.string_input_producer(['./NR-ER/NR-ER-train/names_labels.csv'],shuffle=False)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


record_defaults = [["Null"],[1]] 
example, label = tf.decode_csv(value, record_defaults=record_defaults) # 
train_labl=list()

#print("list",labelarr)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(7697):
        e_val, l_val = sess.run([example, label])
        train_labl.append(l_val)
        #print (l_val)
	
    coord.request_stop()
    coord.join(threads)


data = np.load('./NR-ER/NR-ER-train/names_onehots.npy')
data = data.item()
train_smile = data["onehots"]

ones = []
trues = []
for j in range(0,7697):	
	if train_labl[j] == 1:
		ones.append(train_labl[j])
		trues.append(train_smile[j])
		
ones = np.asarray(ones)
ones = ones[np.newaxis]
trues = np.asarray(trues)
b = np.zeros((937,2))
b[np.arange(937),ones] = 1
# print(trues.shape)
# print(b)

train_labl = np.asarray(train_labl)
labelarray=train_labl[np.newaxis]
final = np.zeros((7697,2))
final[np.arange(7697),labelarray] = 1
train_smiles = np.vstack((train_smile, trues))
train_lables = np.vstack((final, b))
for k in range(6):
 	train_smiles = np.vstack((train_smiles, trues))
 	train_lables = np.vstack((train_lables, b))


filename_queue = tf.train.string_input_producer(['./NR-ER/NR-ER-test/names_labels.csv'],shuffle=False)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


record_defaults = [["Null"],[1]] 
example, label = tf.decode_csv(value, record_defaults=record_defaults) 
test_labl=[]
#print("list",labelarr)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(265):
        e_val, l_val = sess.run([example, label])
        test_labl.append(l_val)
        #print (l_val)
	
    coord.request_stop()
    coord.join(threads)



test_lab= np.asarray(test_labl)
test_lab=test_lab[np.newaxis]
test_lable = np.zeros((265,2))
test_lable[np.arange(265),test_lab] = 1
data = np.load('./NR-ER/NR-ER-test/names_onehots.npy')
data = data.item()
test_smile = data["onehots"]

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
	#print(y_pre,v_ys)
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    #print(len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


#batchInput, batchLabels = get_batch_data(mnist.train.images, mnist.train.labels, batchSize)

#inputs
xs = tf.placeholder(tf.float32,[None,72,398])/255. #72*398
ys = tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,72,398,1])


##conv1 layer##
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #output size
h_pool1 = max_pool_2x2(h_conv1)

# ##conv1 layer##
# W_conv2 = weight_variable([5,5,32,64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #output size
# h_pool2 = max_pool_2x2(h_conv2) 

## fc1 layer ##
W_fc1 = weight_variable([36*199*32,1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1,[-1,36*199*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#print('lll')
for i in range(3000):
    train_batch_xs,train_batch_ys = next_batch(100,train_smiles, train_lables)
    #print(i)
    #print(train_batch_ys)
    
    #train_batch_xs.reshape(100,28656)
    
    #print(train_batch_xs.shape,train_batch_ys.shape)
    #test_batch_xs,test_batch_ys = next_batch(test_smile, test_lable, batchSize) #print(batch_xs)
    #print(sess.run(b_conv1))
    _, guess,cross = sess.run([train_step,prediction,cross_entropy], feed_dict={xs: train_batch_xs, ys: train_batch_ys, keep_prob: 0.5})
    #sess.run(train_step, feed_dict={xs: train_batch_xs, ys: train_batch_ys, keep_prob: 0.5})
    #print(cross)
    #print(sess.run(h_pool1.shape))
    #print(test_smile[:265].shape, test_lable[:265].shape)
    if i % 50 == 0:
        print(compute_accuracy(
            test_smile[:265], test_lable[:265]))

