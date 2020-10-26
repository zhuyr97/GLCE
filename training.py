import random
import os
import re
import time
import numpy as np
import tensorflow as tf
from scipy import io
from Histogram_loss import Diff_Dist_tensor_with_diff
import matplotlib.pyplot as plt
from loss import local_loss,compute_L1_loss,compute_L2_loss,compute_gradient_loss
from inference import inference

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.reset_default_graph()


############################################################################
batch_size = 1  # batch size
num_patch = 3.6e5
iterations = int(num_patch//batch_size)        # iteration
lr0 = 1e-4  # learning rate
num_channels = 1 # image channels
nbins = 255 #256
VALUE_RANGE = [0.0, 255.0]
############################################################################

save_model_path = '/gdata/zhuyr/ImageEnhance/model_training/' # path of saved model
save_path_full = os.path.join(save_model_path, 'model')
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

def _parse_function(filename, label):  
     
  image_string = tf.read_file(filename)  
  image_decoded = tf.image.decode_png(image_string, channels=num_channels)  
  input = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  
  image_string = tf.read_file(label)  
  image_decoded = tf.image.decode_png(image_string, channels=num_channels)  
  label = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  return input, label

def savemodel(save_model_path,training_error):
    return io.savemat(save_model_path + 'training_error.mat', {'training_error': training_error})    
    


if __name__ == '__main__':
   learning_rate = tf.placeholder(tf.float32, [])
 
   ################## training data #################################### 
   input_path = '/gdata/zhuyr/ImageEnhance/input/'    # the path of input images
   gt_path =  '/gdata/zhuyr/ImageEnhance/gt/'     # the path of ground truth

   filename = os.listdir(input_path)
   for i in range(len(filename)):
      filename[i] = input_path + filename[i]
      
   labelfile = os.listdir(gt_path)    
   for i in range(len(labelfile)):
       labelfile[i] = gt_path + labelfile[i] 
    
   filename_tensor = tf.convert_to_tensor(filename, dtype=tf.string)  
   labels_tensor = tf.convert_to_tensor(labelfile, dtype=tf.string)  

   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.shuffle(buffer_size = 100 * batch_size)
   dataset = dataset.prefetch(buffer_size = 10 * batch_size)
   dataset = dataset.batch(batch_size).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   input, label = iterator.get_next()    
   out,mask=inference(input)

   loss1 = compute_L2_loss(label, out)+local_loss(label,out)*5
   hist_label,diff_label = Diff_Dist_tensor_with_diff(label)
   hist_out,diff_out = Diff_Dist_tensor_with_diff(out)

   loss2 = compute_L1_loss(hist_label,hist_out)*2e3
   loss3 = compute_gradient_loss(label, out)*0.5
   loss = loss1 +loss2+loss3

   all_vars = tf.trainable_variables() 

   var_ = [var for var in all_vars if 'CNN' in var.name]
   print("Total parameters' number: %d" %(np.sum([np.prod(v.get_shape().as_list()) for v in var_])))    
   g_optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_) # optimizer for the generator

   saver = tf.train.Saver(var_list=var_, max_to_keep=5)
   
   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True
   config.gpu_options.per_process_gpu_memory_fraction = 0.95
   
   init =  tf.group(tf.global_variables_initializer(), 
                         tf.local_variables_initializer())
   with tf.Session(config=config) as sess:
        sess.run(init)
        tf.get_default_graph().finalize()
        if tf.train.get_checkpoint_state(save_model_path):   # load previous trained model 
           ckpt = tf.train.latest_checkpoint(save_model_path)
           saver.restore(sess, ckpt)  
           ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
           start_point = int(ckpt_num[-1])   
           print("Load success")
           current = start_point + 2
           tmp = io.loadmat(save_model_path + 'training_error.mat')
           training_error = tmp["training_error"]   
           
        else:  # re-training when no model found
           current = 0  
           print("re-training")
           training_error =  np.zeros([1,int(iterations)])
           total_loss = 0.
           
        #check_data, check_labelB =  sess.run([inputs, label])
        
        start = time.time()  
        total_loss = 0.     
        num=0

        decayed_learning_rate = lr0 
        for i in range(current,iterations):

              
            _,Training_Loss,middle_loss1,middle_loss2,middle_loss3 = sess.run([g_optim,loss,loss1,loss2,loss3],feed_dict={learning_rate: decayed_learning_rate})                
                                    
            num += 1      
            total_loss += Training_Loss 
            training_error[0,i] =  total_loss/num                

               
            if np.mod(i+1,100) == 0 and i != iterations:          
                     end = time.time()
                     print ('%d / %d iters, Loss  = %.4f,middle_loss1 = %.4f, moddile-Loss2  = %.4f,moddile-Loss3  = %.6f,Lr  = %.6f, runtime = %.1f s' % 
                           (i+1, iterations, Training_Loss,middle_loss1,middle_loss2,middle_loss3,decayed_learning_rate, (end-start) ) )             
                     start = time.time()  
                     
                     if np.mod(i+1,1000) == 0:                      
                         saver.save(sess, save_path_full, global_step = (i+1), write_meta_graph=False)
                         savemodel(save_model_path,training_error)
        print('Training is finished.')
        
        