import datetime
from tqdm import tqdm
import os
import tensorflow as tf

from losses import generative_loss, discriminative_loss, pretrained_loss
import dataloader
import config
from models import GENERATOR, DISCRIMINATOR

# Initialize models
gModel = GENERATOR()
generator = gModel()

dModel = DISCRIMINATOR()
discriminator = dModel()

pre_epochs = config.pretrained_epochs
epochs = config.epochs

log_ = config.logs

dataset = dataloader.dataset

# pre-train first
if log_: 
  log_dir = config.logs_path
  summary_writer = tf.summary.create_file_writer(
    log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

pretrain_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def pre_step(L_target, ab_target, step):
  with tf.GradientTape() as preTape:
    L = tf.repeat(L_target, repeats = 3, axis = 3)
    ab_predict = generator(L, training = True)
    l1 = pretrained_loss(ab_target, ab_predict)
  
  grads = preTape.gradient(l1, generator.trainable_variables)
  pretrain_opt.apply_gradients(zip(grads,
                                   generator.trainable_variables))
  
  if log_:
    with summary_writer.as_default():
      tf.summary.scalar('L1_pretrained_Loss', l1, step = step//10)
    
def pre_fit(dataset, epochs):
  for epoch in range(epochs):
    for idx, (L, ab) in tqdm(dataset.enumerate()):
      pre_step(L, ab, idx)
    
    gen_ckpt_dir = config.pretrained_weight_path_
    gen_ckpt_name = 'pre_generator-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    generator.save_weights(os.path.join(gen_ckpt_dir, gen_ckpt_name))
   
  
# GANs train
if log_: 
  log_dir = config.logs_path
  summary_writer = tf.summary.create_file_writer(
    log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

gOpt = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
dOpt = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

@tf.function
def train_step(Limg, ab_target, step):
  with tf.GradientTape() as gTape, tf.GradientTape() as dTape:
    L = tf.repeat(Limg, 3, axis = 3)
    ab_predict = generator(L, training = True) # ab Generative image

    d_predict = discriminator([Limg, ab_predict], training = True) # Discriminator output of predict
    d_target = discriminator([Limg, ab_target], training = True) # Discriminator output of target

    # Discriminative Loss
    d_loss = discriminative_loss(d_target, d_predict)

    # Generative loss
    g_loss, g_gan_loss = generative_loss(ab_target, ab_predict, d_predict)

  gGradients = gTape.gradient(g_loss, generator.trainable_variables)
  dGradients = dTape.gradient(d_loss, discriminator.trainable_variables)

  gOpt.apply_gradients(zip(gGradients,
                           generator.trainable_variables))
  dOpt.apply_gradients(zip(dGradients,
                           discriminator.trainable_variables))
  
  if log_:
    with summary_writer.as_default():
      tf.summary.scalar('Total Gen loss', g_loss, step = step//10)
      tf.summary.scalar('Gan loss', g_gan_loss, step = step//10)
      tf.summary.scalar('Total Disc loss', d_loss, step = step//10)
      # tf.summary.scalar('Positive Disc loss', pos_d_loss, step = step//10)
      # tf.summary.scalar('Negative Disc loss', neg_d_loss, step = step//10)

def fit(dataset, epochs):
  for epoch in range(epochs):
    for idx, (L, ab) in tqdm(dataset.enumerate()):
      train_step(L, ab, idx)

    # save generator weights and discriminator weights after each epochs
    gen_ckpt_dir = config.gen_weight_path_
    gen_ckpt_name = 'generator-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    generator.save_weights(os.path.join(gen_ckpt_dir, gen_ckpt_name))

    dis_ckpt_dir = config.dis_weight_path_ 
    dis_ckpt_name = 'discriminator-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    discriminator.save_weights(os.path.join(dis_ckpt_dir, dis_ckpt_name))
    
def main():
  print('Pre-training')
  pre_fit(dataset, pre_epochs)
  
  print('Running GANs training')
  fit(dataset, epochs)
  
if __name__ == '__main__':
  main()
