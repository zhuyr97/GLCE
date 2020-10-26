import os
import cv2
from skimage import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from inference import inference
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()

input_path = './testing_grayimgs/'# the path of testing images
result_path ='./testing_gray_results/' # the path of saving images
save_model_path= './pre-trained_model/' 
model =save_model_path +"model"
if not os.path.exists(result_path):
    os.mkdir(result_path)
def _parse_function(filename):  
    image_string = tf.read_file(filename)   
    image_decoded = tf.image.decode_png(image_string, channels=1)
    images = tf.cast(image_decoded, tf.float32)/255.0
    return images 

if __name__ == '__main__':

    imgName = os.listdir(input_path)
    filename = os.listdir(input_path)    
    num_img = len(filename) 

    print('-' * 20)
    for i in range(num_img):
        filename[i] = input_path + filename[i]

    filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
    dataset = dataset.map(_parse_function)    
    dataset = dataset.prefetch(buffer_size=1 * 10)
    dataset = dataset.batch(1).repeat()  
    iterator = dataset.make_one_shot_iterator()

    input = iterator.get_next()
    output,_=inference(input)
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
            final_output, ori = sess.run([enhanced,input])        
            final_output = np.uint8(final_output* 255.)
            print(i,np.array(final_output).shape)
            index = imgName[i].rfind('.')
            name = imgName[i][:index]
            io.imsave(result_path + name +'.png', final_output)   
        time_end=time.time()
        print('All finished')
        print('totally cost',time_end-time_start)
        print('mean cost',(time_end-time_start)/num_img)
    sess.close()