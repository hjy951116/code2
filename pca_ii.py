import tensorflow as tf
import os
from glob import glob
import scipy.misc
import numpy as np
import sys, getopt

def get_image(image_path):
    image = scipy.misc.imread(image_path)#.astype(np.float)
    return np.array(image) # 0~255 to -1.0~1.0

def main(argv):
    if len(sys.argv) != 3:
        print ('python3 main.py <input_dir> <output_dir>')
        sys.exit(0)
        
    f_path = sys.argv[1]
    o_path = sys.argv[2]
        
    if not os.path.isdir(f_path):
        print ('wrong input data directory')
        sys.exit(1)
        
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    
    types = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
    data = []
    for files in types:
        data.extend(glob(os.path.join(f_path, files)))
    image = [get_image(x) for x in data]
    
    cnt = 0
    sess = tf.Session()
    for i in image:
        ii_images, alpha = pca_ii(i)
        ii_image, angle = sess.run([ii_images, alpha])
        o_fullpath = os.path.join(o_path, os.path.basename(data[cnt]))
        print (os.path.basename(data[cnt]), angle * 180 / 3.1415)
        scipy.misc.imsave(o_fullpath, ii_image)
        cnt += 1
    
def pca_ii(ori_image):
    # Resize particular size
    ori_image = tf.stack(ori_image) # if dim 3 -> 4
    ori_image = tf.expand_dims(ori_image, [0])
    ori_image = tf.cast(ori_image, tf.float32)
    ori_image += 1.0 # Prevent zero divide

    # Extract uniform k sample
    ksizes = [1, 1, 1, 1] # 1x1 pixel
    strides = [1, 10, 10, 1] # 4
    rates = [1, 1, 1, 1]

    sample_pixel = tf.extract_image_patches(ori_image, ksizes, strides, rates, padding='VALID')
    sample_pixel = tf.squeeze(sample_pixel, [0])

    num_sample = sample_pixel.get_shape()[0].value * sample_pixel.get_shape()[1].value
    sample_pixel = tf.reshape(sample_pixel, [num_sample, 1, 1, 3])

    pixel_R = sample_pixel[:,0,0,0]
    pixel_G = sample_pixel[:,0,0,1]
    pixel_B = sample_pixel[:,0,0,2]

    geoM = tf.pow(pixel_R * pixel_G * pixel_B, 1.0/3.0)

    ch_r = tf.reshape(tf.log(pixel_R / geoM), [1, num_sample])
    ch_b = tf.reshape(tf.log(pixel_B / geoM), [1, num_sample])

    X = tf.concat([ch_r, ch_b], 0)
    [U, S, V] = tf.svd(X)

    vec = S[:,0]
    alpha = tf.abs(tf.atan(-vec[0] / vec[1]))

    tmp = tf.squeeze(ori_image, [0])
    pixel_R = tmp[:,:,0]
    pixel_G = tmp[:,:,1]
    pixel_B = tmp[:,:,2]

    geoM = tf.pow(pixel_R * pixel_G * pixel_B, 1.0/3.0)

    num_pixel = ori_image.get_shape()[1].value * ori_image.get_shape()[2].value
    ch_r = tf.log(pixel_R / geoM)
    ch_b = tf.log(pixel_B / geoM)
    
    ii_image = ch_r * tf.cos(alpha) + ch_b * tf.sin(alpha)
    return ii_image, alpha


if __name__ == '__main__':
    main(sys.argv[1:])