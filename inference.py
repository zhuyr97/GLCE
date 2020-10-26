import tensorflow as tf
import numpy as np

def Pre_sub(input1):
	min = tf.reduce_min(input1)
	max = tf.reduce_max(input1)
	input_new = tf.div(tf.subtract(input1,min) ,tf.subtract(max,min))
	return input_new  
def inference(ori ,num=32):
	imgHE = Pre_sub(ori)
	images = tf.reshape(imgHE, shape=[1, tf.shape(ori)[1], tf.shape(ori)[2], 1])   

	a = tf.ones_like(images)
	b = images
	c = b*images
	d = c*images
	bias = tf.concat([a,b,c,d],-1)

	with tf.variable_scope('CNN',  reuse=tf.AUTO_REUSE):  
		with tf.variable_scope('Global'):
			imagesdown = tf.image.resize_images(images, size=[256,256],method=1)     
		
			conv11 = tf.layers.conv2d(imagesdown, num, 3, padding='valid', activation=tf.nn.relu)
			conv12 = tf.layers.conv2d(conv11,  num, 3, padding='valid', activation=tf.nn.relu)     
			POOL1 = tf.nn.max_pool(conv12,[1,2,2,1],[1,2,2,1],padding="VALID")
		
			conv21 = tf.layers.conv2d(POOL1,  num*2, 3, padding='valid', activation=tf.nn.relu)
			conv22 = tf.layers.conv2d(conv21, num*2, 3, padding='valid', activation=tf.nn.relu)   
			POOL2 = tf.nn.max_pool(conv22,[1,2,2,1],[1,2,2,1],padding="VALID")
		
			conv31 = tf.layers.conv2d(POOL2, num*4, 3, padding='valid', activation=tf.nn.relu)
			conv32 = tf.layers.conv2d(conv31,num*4, 3, padding='valid', activation=tf.nn.relu)   
			POOL3 = tf.nn.max_pool(conv32,[1,2,2,1],[1,2,2,1],padding="VALID")

			conv41 = tf.layers.conv2d(POOL3, num*8, 3, padding='valid', activation=tf.nn.relu)
			conv_gap = tf.reduce_mean(conv41, [1, 2], name='GAP', keep_dims=False)
		

			h = tf.layers.dense(conv_gap, 12*12)
			h = tf.nn.relu(h)
			h = tf.layers.dense(h, 6*6)
			h = tf.nn.relu(h)      

			h = tf.layers.dense(h,4)
			matrix =  tf.reshape(h, shape=[1,1,1,4]) 
			global_res = matrix * bias
			global_res = tf.reduce_sum(global_res,[3],keep_dims = True)
		with tf.variable_scope('local'):
			conv_11 = tf.layers.conv2d(images, num, 3, padding='same', activation=tf.nn.relu)
			conv_12 = tf.layers.conv2d(conv_11, num, 3, padding='same', activation=tf.nn.relu)     
		
			conv_21 = tf.layers.conv2d(conv_12,  num, 3, padding='same', activation=tf.nn.relu)
			conv_22 = tf.layers.conv2d(conv_21, num, 3, padding='same', activation=tf.nn.relu)           
		
			conv_31 = tf.layers.conv2d(conv_22, num, 3, padding='same', activation=tf.nn.relu)
			conv_32 = tf.layers.conv2d(conv_31,num, 3, padding='same', activation=tf.nn.relu)   

			conv_41 = tf.layers.conv2d(conv_32, num, 3, padding='same', activation=tf.nn.relu)
			matrix_ = tf.layers.conv2d(conv_41, 4, 3, padding='same', activation=tf.nn.relu)
			
			local_res = matrix_ * bias
			local_res = tf.reduce_sum(local_res,[3],keep_dims = True)
		out = images + global_res +local_res
		mask =matrix_

	return  out,mask