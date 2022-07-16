import tensorflow as tf

def generative_loss(target, predict, discriminator_output_of_predict, LAMBDA = 100.0):
  l1_loss = tf.reduce_mean(tf.abs(predict - target))

  generative_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.ones_like(discriminator_output_of_predict),
                                                                           discriminator_output_of_predict)
  total_loss = generative_loss + LAMBDA*l1_loss
  return total_loss, generative_loss

def discriminative_loss(target, predict):
  posLoss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.ones_like(target),
                                                                   target)
  negLoss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.zeros_like(predict),
                                                                   predict)
  return posLoss + negLoss

def pretrained_loss(target, predict):
  l1 = tf.reduce_mean(tf.abs(target - predict))
  return l1
