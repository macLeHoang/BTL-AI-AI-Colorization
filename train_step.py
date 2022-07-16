import tensorflow as tf

from losses import generative_loss, discriminative_loss, pretrained_loss
from data

// pre-train first
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
  
  with summary_writer.as_default():
    tf.summary.scalar('L1_pretrained_Loss', l1, step = step//10)
    
def pre_fit(dataset, epochs):
  for epoch in range(epochs):
    for idx, (L, ab) in tqdm(dataset.enumerate()):
      pre_step(L, ab, idx)
    
    gen_ckpt_dir = '/content/gdrive/MyDrive/AI_color_weights/Generator_v2'
    gen_ckpt_name = 'pre_generator-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    generator.save_weights(os.path.join(gen_ckpt_dir, gen_ckpt_name))

