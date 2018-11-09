#!/usr/bin/env
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
zeros = []
falses = []
for j in range(0,7697): 
    if train_labl[j] == 0:
        zeros.append(train_labl[j])
        falses.append(train_smile[j])
        
zeros = np.asarray(zeros)
zeros = zeros[np.newaxis]
falses = np.asarray(falses) 
a = np.zeros((937,2))
a[np.arange(937),ones] = 1


# train_labl = np.asarray(train_labl)
# labelarray=train_labl[np.newaxis]
# final = np.zeros((7697,2))
# final[np.arange(7697),labelarray] = 1
# train_smiles = np.vstack((train_smile, trues))
# train_lables = np.vstack((final, b))
# for k in range(6):
#     train_smiles = np.vstack((train_smiles, trues))
#     train_lables = np.vstack((train_lables, b))


train_smiles10 = falses[0:937]
train_labels10 = zeros[0:937]
train_smiles20 = falses[937:1847]
train_labels20 = zeros[937:1847]
train_smiles30 = falses[1847:2811]
train_labels30 = falses[1847:2811]
train_smiles40 = zeros[2811:3748]
train_labels40 = falses[2811:3748]
train_smiles50 = zeros[3748:4685]
train_labels50 = falses[3748:4685]
train_smiles60 = zeros[4685:5622]
train_labels60 = falses[4685:5622]
train_smiles70 = zeros[5622:6559]
train_smiles70 = falses[5622:6559]

train_smiles[1] = np.vstack((train_smiles10, trues))
train_lables[1] = np.vstack((train_lables10, b))
train_smiles[2] = np.vstack((train_smiles20, trues))
train_lables[2] = np.vstack((train_lables20, b))
train_smiles[3] = np.vstack((train_smiles30, trues))
train_lables[3] = np.vstack((train_lables30, b))
train_smiles[4] = np.vstack((train_smiles40, trues))
train_lables[4] = np.vstack((train_lables40, b))
train_smiles[5] = np.vstack((train_smiles50, trues))
train_lables[5] = np.vstack((train_lables50, b))
train_smiles[6] = np.vstack((train_smiles60, trues))
train_lables[6] = np.vstack((train_lables60, b))
train_smiles[7] = np.vstack((train_smiles70, trues))
train_lables[7] = np.vstack((train_lables70, b))


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

# def compute_accuracy(v_xs,v_ys):
# 	global prediction
# 	y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
# 	y_pred = tf.argmax(y_pre,1)
# 	y_true = tf.argmax(v_ys,1)
# 	print(y_pred,y_true)
# 	recall1,recall1_op = tf.metrics.recall(y_true,y_pred)
# 	#recall2, recall2_op = tf.metrics.recall(V_ys,y_pre)
	
# 	#print(y_pre,v_ys)
# 	# correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
# 	# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 	result = sess.run([recall1,recall1_op], feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
# 	#print(recall1)
# 	return result

def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
  #print(y_pre[:10])
  y_pred = tf.argmax(y_pre, 1)
  actuals = tf.argmax(v_ys, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(y_pred)
  zeros_like_predictions = tf.zeros_like(y_pred)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(y_pred, ones_like_predictions)
      ), 
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(y_pred, zeros_like_predictions)
      ), 
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(y_pred, ones_like_predictions)
      ), 
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(y_pred, zeros_like_predictions)
      ), 
      "float"
    )
  )

  tp, tn, fp, fn = sess.run(
      [tp_op, tn_op, fp_op, fn_op], 
      feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1}
    )

  tpr = float(tp)/(float(tp) + float(fn))
  fpr = float(tn)/(float(tn) + float(fp))

  return 1/2 * (tpr + fpr)

  #accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

  # recall = tpr
  # precision = float(tp)/(float(tp) + float(fp))

  # f1_score = (2 * (precision * recall)) / (precision + recall)


def weight_variable(name,shapen):
	 initial = tf.truncated_normal(shape,stddev=0.01)
	 return tf.Variable(initial)
  #initial = tf.get_variable(name,shape=shapen,
           #initializer=tf.contrib.layers.xavier_initializer())
  return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
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


##conv1 layer##SAME
W_conv1 = weight_variable("W_conv1",[72,3,1,64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,73,1,1],padding='SAME')+b_conv1) #output size 72*398*64
h_pool1 = tf.nn.max_pool(h_conv1,poolsize=[1,1,2,1],strides=[1,1,2,1],padding='SAME') #72*198*64

# ##conv1 layer##
W_conv2 = weight_variable("W_conv2",[5,5,64,128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2) #output size 72*198*128
h_pool2 = tf.nn.max_pool(h_conv2,poolsize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #36*99*128
           

## fc1 layer ##
W_fc1 = weight_variable("W_fc1",[36*99*128,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,36*99*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable("W_fc2",[1024, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)




if True:
    # For each of the neural networks.
    for j in range(6):
        print("Neural network: {0}".format(i))
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
          for i in range(2000):
            train_batch_xs,train_batch_ys = next_batch(50,train_smiles[j+1], train_lables[j+1])
            #print(i)
            #print(train_batch_ys)
            
            #train_batch_xs.reshape(100,28656)
            
            #print(train_batch_xs.shape,train_batch_ys.shape)
            #test_batch_xs,test_batch_ys = next_batch(test_smile, test_lable, batchSize) #print(batch_xs)
            #print(sess.run(b_conv1))
            
            _, guess= sess.run([train_step,prediction], feed_dict={xs: train_batch_xs, ys: train_batch_ys, keep_prob: 0.5})
            
        #sess.run(train_step, feed_dict={xs: train_batch_xs, ys: train_batch_ys, keep_prob: 0.5})
        #print(cross)
        #print(sess.run(h_pool1.shape))
        #print(test_smile[:265].shape, test_lable[:265].shape)
            if i % 100 == 0:
                cross1 = sess.run([cross_entropy], feed_dict={xs: test_smile[:265], ys: test_lable[:265], keep_prob: 1})
                cross= sess.run([cross_entropy], feed_dict={xs: train_batch_xs, ys: train_batch_ys, keep_prob: 1})
                print(cross1)
                print(compute_accuracy(
                    test_smile[:265], test_lable[:265]))
                print(cross)
                print(compute_accuracy(
                    train_batch_xs, train_batch_ys))
        # Create a random training-set. Ignore the validation-set.
       

        # Initialize the variables of the TensorFlow graph.
        

        # Optimize the variables using this training-set.
        

        # Save the optimized variables to disk.
        # saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        print()
#print('lll')

        # print(compute_accuracy(
        #     train_batch_xs,train_batch_ys))
# saver = tf.train.Saver()
# saver.save(sess, './my_model', global_step = 1)

