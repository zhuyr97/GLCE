import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def Diff_Dist_tensor(x, bins=256, min=0.0, max=1.0):
    n_batch = tf.shape(x)[0]
    row_m = tf.shape(x)[1]
    row_n = tf.shape(x)[2]
    channels = tf.shape(x)[3]

    delta = (max - min) / bins
    BIN_Table = np.arange(0, bins, 1)  # .astype(np.float64)
    BIN_Table = BIN_Table * delta

    temp = tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]], dtype=tf.float32, name='temp')
    temp1 = tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]], dtype=tf.float32, name='temp1')

    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim]  # h_r
        h_r_sub_1 = BIN_Table[dim - 1]  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1]  # h_(r+1)
        # 中间计算都是tensor进行的
        h_r = tf.convert_to_tensor(h_r, dtype=tf.float32, name='h_r')
        h_r_sub_1 = tf.convert_to_tensor(h_r_sub_1, dtype=tf.float32, name='h_r_sub_1')
        h_r_plus_1 = tf.convert_to_tensor(h_r_plus_1, dtype=tf.float32, name='h_r_plus_1')

        h_r_temp = h_r * temp
        h_r_sub_1_temp = h_r_sub_1 * temp
        h_r_plus_1_temp = h_r_plus_1 * temp

        mask_sub = tf.where(tf.greater(h_r_temp, x) & tf.greater(x, h_r_sub_1_temp), temp, temp1)
        mask_plus = tf.where(tf.greater(x, h_r_temp) & tf.greater(h_r_plus_1_temp, x), temp, temp1)

        temp_mean1 = tf.reduce_mean(tf.reshape(((x - h_r_sub_1) * mask_sub), (n_batch, channels, -1)), axis=-1)
        temp_mean2 = tf.reduce_mean(tf.reshape(((h_r_plus_1 - x) * mask_plus), (n_batch, channels, -1)), axis=-1)

        if dim == 1:
            temp_mean = tf.add(temp_mean1, temp_mean2)
            temp_mean = tf.expand_dims(temp_mean, -1)
        else:
            temp_mean_temp = tf.add(temp_mean1, temp_mean2)
            temp_mean_temp = tf.expand_dims(temp_mean_temp, -1)
            temp_mean = tf.concat([temp_mean, temp_mean_temp], axis=-1)
    return temp_mean

def Diff_Dist_tensor_with_diff(x,bins=256, min=0.0, max=1.0):

    n_batch=tf.shape(x)[0]
    row_m = tf.shape(x)[1]
    row_n = tf.shape(x)[2]
    channels = tf.shape(x)[3]
 
    delta = (max - min) / bins
    BIN_Table = np.arange(0, bins, 1)  # .astype(np.float64)
    BIN_Table = BIN_Table * delta

    zero = tf.constant([[[0.0]]])
    temp = tf.ones([tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]],dtype=tf.float32,name='temp')
    temp1 =tf.zeros([tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]],dtype=tf.float32,name='temp1')
    for dim in range(1, bins-1 , 1):
        h_r = BIN_Table[dim]  # h_r
        h_r_sub_1 = BIN_Table[dim - 1]  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1]  # h_(r+1)
        #中间计算都是tensor进行的
        h_r = tf.convert_to_tensor(h_r,dtype=tf.float32,name='h_r')
        h_r_sub_1= tf.convert_to_tensor(h_r_sub_1,dtype=tf.float32,name='h_r_sub_1')
        h_r_plus_1 = tf.convert_to_tensor(h_r_plus_1,dtype=tf.float32,name='h_r_plus_1')

        h_r_temp= h_r*temp  
        h_r_sub_1_temp = h_r_sub_1*temp
        h_r_plus_1_temp = h_r_plus_1 *temp

        mask_sub = tf.where(tf.greater(h_r_temp, x)&tf.greater(x,h_r_sub_1_temp),temp,temp1)
        mask_plus = tf.where(tf.greater(x,h_r_temp)&tf.greater(h_r_plus_1_temp,x),temp,temp1)

        temp_mean1 = tf.reduce_mean(tf.reshape(((x - h_r_sub_1) * mask_sub),(n_batch, channels,-1)), axis=-1)
        temp_mean2 = tf.reduce_mean(tf.reshape(((h_r_plus_1 - x) * mask_plus),(n_batch, channels, -1)), axis=-1)
        
        if dim ==1:
            temp_mean = tf.add(temp_mean1, temp_mean2)
            temp_mean =tf.expand_dims(temp_mean,-1)  #[1,1,1]
        else :
            if dim != bins-2:
                temp_mean_temp = tf.add(temp_mean1,temp_mean2)
                temp_mean_temp = tf.expand_dims(temp_mean_temp,-1)
                temp_mean = tf.concat([temp_mean,temp_mean_temp],axis=-1)
            else:
                zero = tf.concat([zero,temp_mean], axis=-1)
                temp_mean_temp = tf.add(temp_mean1, temp_mean2)
                temp_mean_temp = tf.expand_dims(temp_mean_temp, -1)
                temp_mean = tf.concat([temp_mean, temp_mean_temp], axis=-1)

    #assert temp_mean_last.get_shape() == temp_mean.get_shape()
    diff = tf.abs(temp_mean-zero)
    return temp_mean,diff
