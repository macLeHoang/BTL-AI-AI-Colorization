import argparse
from PIL import Image
import numpy as np
import os
from skimage.color import lab2rgb

import config
from models import GENERATOR

parser = argparse.ArgumentParser(description='Configuration')

parser.add_argument('-w', '--weight_path', type = str, help = 'Path to weights file', default = None, metavar='')
parser.add_argument('-i', '--img_path', type = str, help = 'Path to image', required=True, metavar='')
parser.add_argument('-s', '--size', type = tuple, help = 'Output image size - (Width, Height)', default = (256, 256), metavar='')
parser.add_argument('-st', '--store', type = str, help = 'Store path', default = None, metavar = '')
# parser.add_argument('-k', '--keep_dim', action = 'store_true', help = '')
args = parser.parse_args()


gen = GENERATOR(False)
model_ = gen()
default_weight_ = config.weight_

def colored():
  SIZE = args.size
  w = int(SIZE[0] / 32) * 32
  h = int(SIZE[1] / 32) * 32
  img = Image.open(args.img_path).resize((w, h)).convert('L')
  
  img = np.array(img)[..., np.newaxis] / 255. * 2 - 1
  img_ = np.repeat(img, 3, axis = -1)[np.newaxis, ...] 
  ab = model_(img_, training = False)
  
  img = (img+1)*50
  ab = ab*110
  lab = np.concatenate([img, ab[0].numpy()], axis = -1)
  rgb = lab2rgb(lab)
  
  name_ = os.path.split(args.img_path)[-1]
  store_ = name_.split('.')[0] + '_gen.jpg'
  
  rgb = Image.fromarray(np.uint8(rgb*255)).save(store_)

def main():
  if args.weight_path is None:
    if default_weight_ is None:
      print('Please enter a weight path or setup default weight path in config file !!!')
      return
    else:
      model_.load_weights(default_weight_)
  else:
    model_.load_weights(args.weight_path)
  colored()
  
if __name__ == '__main__':
  main()
  


