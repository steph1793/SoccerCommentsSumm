import tensorflow as tf
import time
import numpy as np


def train_model(model, dataset, params, ckpt, ckpt_manager, out_file):
  
  optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'], initial_accumulator_value=params['adagrad_init_acc'], clipnorm=params['max_grad_norm'])
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')
  
  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 1))
    dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens # we have to make sure no empty abstract is being used otherwise dec_lens may contain null values
    return tf.reduce_mean(loss_)

  def coverage_loss(attentions):
    """ coverage loss computation"""
    covloss = 0
    coverage = tf.zeros_like(tf.unstack(attentions[0]))
    for a in tf.unstack(attentions): # a in an attention vector at time step t
      covloss += tf.reduce_mean(tf.minimum(a, coverage )) 
      coverage += a
    return covloss
  
  @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                               tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                tf.TensorSpec(shape=[params["batch_size"], None, 1], dtype=tf.float32),
                               tf.TensorSpec(shape=[], dtype=tf.int32),
                                tf.TensorSpec(shape=[params["batch_size"], None])))
  def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, cov_vec, batch_oov_len, stats):
    loss = 0

    if not params["coverage"]:
      cov_vec=None

    with tf.GradientTape() as tape:
      enc_hidden, enc_output = model.call_encoder(enc_inp)
      res = model(enc_output, enc_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len, cov_vec)
      predictions = res["final_dists"]
      loss = loss_function(dec_tar, predictions)
      total_loss = loss
      if params["coverage"]:
        cov_loss = coverage_loss(res["attn_weights"])
        total_loss += cov_loss * params["cov_weight"]
    variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables + model.pointer.trainable_variables
    gradients = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    if params["coverage"]:
      return total_loss, loss, cov_loss
    else:
      return total_loss
  
  
  
  try:
    for batch in dataset:
      t0 = time.time()
      cov_vec = tf.zeros(shape=[params["batch_size"], int(batch[0]["enc_input"].shape[1]),1 ], dtype=tf.float32)
      losses = train_step(batch[0]["enc_input"], batch[0]["extended_enc_input"], batch[1]["dec_input"], batch[1]["dec_target"], cov_vec, batch[0]["max_oov_len"], batch[0]["stats"])
      if params["coverage"]:
        print('Step {}, time {:.4f}, Total Loss {:.4f}, loss {:.4f}, cov_loss {:.4f}'.format(int(ckpt.step),
                                                       time.time()-t0,
                                                       losses[0].numpy(), losses[1].numpy(), losses[2].numpy()))
      else:
        print('Step {}, time {:.4f}, Total Loss {:.4f}'.format(int(ckpt.step),
                                                       time.time()-t0,
                                                       losses.numpy()))
      if int(ckpt.step) == params["max_steps"]:
        ckpt_manager.save(checkpoint_number=int(ckpt.step))
        print("Saved checkpoint for step {}".format(int(ckpt.step)))
        break
      if int(ckpt.step) % params["checkpoints_save_steps"] ==0 :
        ckpt_manager.save(checkpoint_number=int(ckpt.step))
        print("Saved checkpoint for step {}".format(int(ckpt.step)))
      ckpt.step.assign_add(1)
      
        
  except KeyboardInterrupt:
    ckpt_manager.save(int(ckpt.step))
    print("Saved checkpoint for step {}".format(int(ckpt.step)))