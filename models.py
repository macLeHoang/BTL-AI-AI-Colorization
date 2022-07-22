import tensorflow as tf
from classification_models.tfkeras import Classifiers

# GENERATOR
def process(input_, nfilters_1 = 1024, nfilters_2 = 512, ksize = (3, 3), strides = 1, last_relu = True):
  x = tf.keras.layers.ZeroPadding2D()(input_)
  x = tf.keras.layers.Conv2D(nfilters_1, ksize, strides)(x)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.ZeroPadding2D()(x)
  x = tf.keras.layers.Conv2D(nfilters_2, ksize, strides)(x)

  if last_relu:
    x = tf.keras.layers.ReLU()(x)
  return x

def decoder(input_, concat, nfilters = 1024, ksize = (1, 1), strides = 1):
  x = tf.keras.layers.Conv2D(nfilters, ksize, strides)(input_)
  x = tf.keras.layers.ReLU()(x)
  out = tf.nn.depth_to_space(x, 2)
  x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(out)
  x = tf.keras.layers.Concatenate()([concat, x])
  x = tf.keras.layers.ReLU()(x)
  return x

class GENERATOR(tf.keras.models.Model):
  def __init__(self, _use_imagenet_weight = True, _training = True):
    super().__init__()
    self.ResNet18, _ = Classifiers.get('resnet18')
    self.use_imagenet_weight = _use_imagenet_weight
    self.training = _training
  
  def __call__(self):
    resnet18 = self.ResNet18(input_shape = (256, 256, 3) if self.training else (None, None, 3), 
                             weights = 'imagenet' if self.use_imagenet_weight else None,
                             include_top = False)
  
    encoder_1 = resnet18.get_layer('bn0').output # encode 128 - shape = (None, 128, 128, 64)
    encoder_2 = resnet18.get_layer('stage2_unit1_bn1').output # encode 64 - shape = (None, 64, 64, 64)
    encoder_3 = resnet18.get_layer('stage3_unit1_bn1').output # encode 32 - shape = (None, 32, 32, 128)
    encoder_4 = resnet18.get_layer('stage4_unit1_bn1').output # encode 16 - shape = (None, 16, 16, 256)

    last_layer = resnet18.layers[-1].output
    bridge = process(last_layer)

    x = decoder(bridge, encoder_4) # shape = (None, 16, 16, 512)
    x = process(x, 512, 512)
    x = decoder(x, encoder_3, 1024) # shape = (None, 32, 32, 384)
    x = process(x, 384, 384)
    x = decoder(x, encoder_2, 768) # shape = (None, 64, 64, 256)
    x = process(x, 256, 256)
    x = decoder(x, encoder_1, 512) # shape = (None, 128, 128, 192)
    x = process(x, 96, 96)
    x = tf.keras.layers.Conv2D(384, (1, 1), 1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.nn.depth_to_space(x, 2) # shape = (256, 256, 96)
    x = tf.keras.layers.Concatenate()([x, resnet18.input[:, :, :, 0:1]]) # shape = (None, 256, 256, 97)
    res = process(x, 97, 97, last_relu = False)
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(2, (1, 1), 1)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    return tf.keras.models.Model(inputs = resnet18.input, outputs = x)

# DISCRIMINATOR
initializer = tf.random_normal_initializer(0.0, 0.02)
def down_sample(input, nums_filters, kernel_size = (4, 4), strides = 2, use_batchnorm = True, **kwags):
  x = input
  x = tf.keras.layers.Conv2D(nums_filters, 
                             kernel_size = kernel_size,
                             strides = strides,
                             kernel_initializer = initializer,
                             use_bias = False,
                             padding = 'same',
                             **kwags)(x)
  if use_batchnorm:
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)  
    
  x = tf.keras.layers.LeakyReLU()(x)
  return x 
  

class DISCRIMINATOR(tf.keras.models.Model):
  def __init__(self):
    super().__init__
    self.cpart= tf.keras.layers.Input(shape = (None, None, 1)) # use as condition input
    self.predict = tf.keras.layers.Input(shape = (None, None, 2))
  
  def __call__(self):
    x = tf.keras.layers.Concatenate(axis = -1)([self.cpart, self.predict]) # shape = None, 256, 256, 3

    x = down_sample(x, 64, use_batchnorm = False) # shape = None, 128, 128, 64
    x = down_sample(x, 128) # shape = None, 64, 64, 128
    x = down_sample(x, 256) # shape = None, 32, 32, 256

    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(x) # shape = None, 34, 34, 256
    x = down_sample(x, 512, strides = 1) # shape = None, 31, 31, 512

    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(x) # shape = None, 33, 33, 512

    # Each pixel in the feature map looks up to 70*70 patch of the origin image
    x = tf.keras.layers.Conv2D(1, kernel_size = (4, 4), 
                               strides = 1, 
                               kernel_initializer = initializer)(x) # shape = None, 30, 30, 1
    return tf.keras.models.Model(inputs = [self.cpart, self.predict], outputs = x)
