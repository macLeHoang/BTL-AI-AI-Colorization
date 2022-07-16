from skimage.color import rgb2lab, lab2rgb
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import random
import config

import tensorflow as tf


BATCH_SIZE = config.BATCH
path = config.path_

# this part is for removing gray images in training dataset
c = 0
for f in tqdm(os.listdir(path)):
  fPath = os.path.join(path, f)
  img = Image.open(fPath)
  if img.mode != 'RGB':
    os.remove(fPath)
    c += 1

print()
print(f'Remove {c} gray images')
print(f'Remain {len(os.listdir(path))} images')


SIZE = (256, 256)
path_ = path + '/*jpg'
dataset = tf.data.Dataset.list_files(path_)

def process(path):
  '''
  This function will mapping image path to L-channel and ab-channels then convert them to tf.Tensor
    :param path: path to image
  '''
  path_ = bytes.decode(path.numpy())

  img = Image.open(path_)
  img = img.resize(SIZE, Image.BICUBIC)
  
  # Slightly augmentation
  randNumber = random.random() # create random number in range [0, 1]
  if randNumber > 0.7: # do augmentation if the number created is greater than 0.5
    anotherRandNumber = random.random() 
    if anotherRandNumber < 0.5:
      img = img.transpose(Image.FLIP_LEFT_RIGHT) # flip vertical
      img = np.array(img)

    elif 0.5 < anotherRandNumber:
      img = img.transpose(Image.FLIP_TOP_BOTTOM) # flip horizontal
      img = np.array(img)

    # elif 0.7 < anotherRandNumber:
    #   alpha = random.randint(-30, 30) 
    #   img = img.rotate(alpha, expand = False) # rotate
    #   img = np.array(img)

    # elif anotherRandNumber > 0.9:
    #   img = np.array(img)
    #   sx = random.uniform(-0.2, 0.2) #create random number in range [-0.2, 0.2]
    #   sy = random.uniform(-0.2, 0.2)
    #   matrix = np.asarray([[1, sx, 0], [sy, 1, 0], [0, 0, 1]])
    #   affine = transform.AffineTransform(matrix)
    #   img = transform.warp(img, affine.params) # shere
  else:
    img = np.array(img)

  labImg = rgb2lab(img)
  lChannel = labImg[:, :, 0:1] / 50.0 - 1 # convert L channel to range [-1, 1]
  abChannels = labImg[:, :, 1:] / 110.0 # convert ab channel to range [-1, 1]

  return tf.convert_to_tensor(lChannel, dtype = tf.float32), \
         tf.convert_to_tensor(abChannels, dtype = tf.float32)

dataset = dataset.map(lambda x: tf.py_function(process, [x], [tf.float32, tf.float32]))
dataset = dataset.batch(BATCH_SIZE)
