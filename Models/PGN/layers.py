import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
  
  
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values, cov_features):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    if not tf.is_tensor(cov_features):
      cov_features = 0
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis) + cov_features))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
  
class Coverage(tf.keras.layers.Layer):
  def __init__(self, attn_units):
    super(Coverage, self).__init__()
    self.cov_filter = tf.keras.layers.Conv1D(attn_units, 1, 1, "same")

  def call(self, cov_vector):
    # cov_vector : [batch_size, enc_len, 1]
    #returns [batch_size, 1, attn_hidden_size]
    a= tf.reduce_sum(self.cov_filter(cov_vector), axis=1, keepdims=True)
    return a

class Decoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, use_stats=False):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)
    
    self.use_stats = False
    if use_stats:
      self.use_stats = True
      self.stat_fc1 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
      self.stat_fc2 = tf.keras.layers.Dense(self.dec_units, activation=tf.keras.activations.relu)
    

  def call(self, x, hidden, enc_output, context_vector, stats=None):
    # enc_output shape == (batch_size, max_length, hidden_size)
    

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    if self.use_stats:
      stat_embed = self.stat_fc2(self.stat_fc1(stats))
      x = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(stat_embed, 1),  x], axis=-1)
    else:
      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      x = tf.concat([tf.expand_dims(context_vector, 1),  x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x, initial_state = hidden)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    out = self.fc(output)

    return x, out, state
  

class Pointer(tf.keras.layers.Layer):
  
  def __init__(self):
    super(Pointer, self).__init__()
    self.w_s_reduce = tf.keras.layers.Dense(1)
    self.w_i_reduce = tf.keras.layers.Dense(1)
    self.w_c_reduce = tf.keras.layers.Dense(1)
    
  def call(self, context_vector, state, dec_inp):
    return tf.nn.sigmoid(self.w_s_reduce(state)+self.w_c_reduce(context_vector)+self.w_i_reduce(dec_inp))