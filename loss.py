import numpy as np
import tensorflow as tf
from Diff_Dist import Diff_Dist_tensor

def compute_L1_loss(pred, gt):
   loss = tf.abs(pred - gt)
   loss = tf.reduce_mean(loss)
   return loss  
   
def compute_L2_loss(pred, gt):
   loss = tf.square(pred - gt)
   loss = tf.reduce_mean(loss)
   return loss  
 
def compute_gradient_loss(img1,img2):  # be used in compute_exclusion_loss
    gradx1 = img1[:, 1:, :, :] - img1[:, :-1, :, :]
    grady1 = img1[:, :, 1:, :] - img1[:, :, :-1, :]

    gradx2 = img2[:, 1:, :, :] - img2[:, :-1, :, :]
    grady2 = img2[:, :, 1:, :] - img2[:, :, :-1, :]

    loss = compute_L1_loss(gradx1,gradx2) + compute_L1_loss(grady1,grady2) 
    return loss
	
def local_loss(label,out):
    #label [1,H,W,1]  gray image; paired
    #out [1,H,W,1]  gray image 
    H = tf.shape(label)[1]//3
    W = tf.shape(label)[2]//3
    
    t =np.random.randint(100,size=5)
    #select different seed 
    t1 =t[0]
    t2 =t[1]
    t3 =t[2]
    
    #the values that needed to crop
    label_value = tf.squeeze(label,axis=0)
    out_value = tf.squeeze(out,axis=0)
    
    #first crop
    label_crop1 = tf.random_crop(label_value,[H,W,1],seed=t1)
    out_crop1 = tf.random_crop(out_value,[H,W,1],seed=t1)
    label_crop1 = tf.expand_dims(label_crop1,0)
    out_crop1 = tf.expand_dims(out_crop1,0)
    #second crop
    label_crop2 = tf.random_crop(out_value,[H,W,1],seed=t2)
    out_crop2 = tf.random_crop(out[0,:,:,:],[H,W,1],seed=t2)
    label_crop2 = tf.expand_dims(label_crop2,0)
    out_crop2 = tf.expand_dims(out_crop2,0)
    #third crop
    label_crop3 = tf.random_crop(label_value,[H,W,1],seed=t3)
    out_crop3 = tf.random_crop(out_value,[H,W,1],seed=t3)
    label_crop3 = tf.expand_dims(label_crop3,0)
    out_crop3 = tf.expand_dims(out_crop3,0)
    
    loss1 = compute_L2_loss(label_crop1,out_crop1)
    loss2 = compute_L2_loss(label_crop2,out_crop2)
    loss3 = compute_L2_loss(label_crop3,out_crop3)
       
    
    return loss1+loss2 +loss3


