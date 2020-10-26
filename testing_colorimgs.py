import os,time
import cv2
from skimage import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from inference import inference

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.reset_default_graph()

input_path = './testing_colorimgs/'# the path of testing images
result_path ='/testing_color_results/' # the path of saving images
save_model_path= './pre-trained_model/' 
model =save_model_path +"model"
if not os.path.exists(result_path):
    os.mkdir(result_path)
imgName = os.listdir(input_path)
filename = os.listdir(input_path)    
num_img = len(filename) 

image = tf.placeholder(tf.float32, shape=(1, None, None, 1))
output,_=inference(image)
output = tf.clip_by_value(output, 0., 1.)
enhanced = output[0,:,:,0]

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth=True  

var_CSC = [var for var in tf.trainable_variables()  if 'CNN' in var.name]     
saver_CSC = tf.train.Saver(var_list = var_CSC)

with tf.Session(config=config) as sess:
	saver_CSC.restore(sess, model)
	print("Load success")

	time_start=time.time()
	for i in range(num_img):
		filename[i] = input_path + filename[i]
		print(filename[i])
		img=cv2.imread(filename[i])
		img_shape = img.shape
		if len(img_shape)==3:
			yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
			Y,U,V = cv2.split(yuv);  
			if np.max(Y) > 1:
				Y = Y/255.0
			Y = np.expand_dims(Y[:,:], axis = -1)
			input_tensor = np.expand_dims(Y[:,:,:], axis = 0)
		else:
			if np.max(img) > 1:
				img = img/255.0
			img = np.expand_dims(img[:,:], axis = -1)
			input_tensor = np.expand_dims(img[:,:,:], axis = 0)
		final_output  = sess.run(enhanced, feed_dict={image: input_tensor})
		if len(img_shape)==3:
			final_output = np.uint8(final_output* 255.)
			img_out = cv2.merge([final_output, U, V]) 
			img_out = cv2.cvtColor(img_out,cv2.COLOR_YUV2BGR)
		else:
			img_out = np.uint8(final_output* 255.)
		index = imgName[i].rfind('.')
		name = imgName[i][:index]
		cv2.imwrite(result_path + name +'.png', img_out) 
	time_end=time.time()
	print('totally cost',time_end-time_start,(time_end-time_start)/num_img)
	print('All finished')
sess.close()   