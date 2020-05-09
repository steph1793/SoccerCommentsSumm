import tensorflow as tf
from utils import _calc_final_dist
from layers import Encoder, BahdanauAttention, Coverage, Decoder, Pointer

class PGN(tf.keras.Model):
  
  def __init__(self, params):
    super(PGN, self).__init__()
    self.params = params
    self.encoder = Encoder(params["vocab_size"], params["embed_size"], params["enc_units"], params["batch_size"])
    self.attention = BahdanauAttention(params["attn_units"])
    if params["coverage"]:
      self.coverage = Coverage(params["attn_units"])
    self.decoder = Decoder(params["vocab_size"], params["embed_size"], params["dec_units"], params["batch_size"], params["use_stats"])
    self.pointer = Pointer()


  def call_encoder(self, enc_inp):
    enc_hidden = self.encoder.initialize_hidden_state()
    enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
    return enc_hidden, enc_output
    
  def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp,  dec_inp, batch_oov_len, cov_vec, stats=None):
    
    predictions = []
    attentions = []
    p_gens = []
    if self.params["coverage"]:
      cov_features = self.coverage(cov_vec)
    else:
      cov_features = None
    context_vector, _ = self.attention(dec_hidden, enc_output, cov_features)

    for t in range(dec_inp.shape[1]):
      dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t],1), dec_hidden, enc_output, context_vector, stats)
      
      if self.params["coverage"]:
        cov_features = self.coverage(cov_vec)
      else:
        cov_features = None
      context_vector, attn = self.attention(dec_hidden, enc_output, cov_features)
      p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
      if self.params["coverage"]:
        cov_vec += attn
      
      attn = tf.squeeze(attn, axis=-1)
      predictions.append(pred)
      attentions.append(attn)
      p_gens.append(p_gen)
    
    final_dists = _calc_final_dist( enc_extended_inp, predictions, attentions, p_gens, batch_oov_len, self.params["vocab_size"], self.params["batch_size"])
    res = {}
    res["final_dists"] = tf.stack(final_dists, 1)
    res["dec_hidden"] = dec_hidden
    if self.params["coverage"] or self.params["mode"] != "train":
      res["cov_vec"] = cov_vec
      res["attn_weights"] = tf.stack(attentions, 1)
    if self.params["mode"] != "train":
      res["context"] = context_vector
      res["p_gens"] = tf.stack(p_gens, 1)
    return res   # predictions_shape = (batch_size, dec_len, vocab_size) with dec_len = 1 in pred mode