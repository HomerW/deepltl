import tensorflow as tf

class LTLOperator(tf.keras.layers.Layer):
  """
  Represents learned LTL operators
  Input: The original trace or a sequence of activations of LTL filters in
         the previous layer
  Output: The result of applying the filters to the sequence
  """
  def __init__(self, num_variables, num_filters, trace_length,
               kernel_regularizer=None, metric=True, **kwargs):
    self.num_filters = num_filters
    self.num_variables = num_variables
    self.trace_length = trace_length
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.metric = metric
    self.slope = tf.Variable(1.0, trainable=False)
    self.alpha = tf.Variable(0.2, trainable=False)
    super(LTLOperator, self).__init__(**kwargs)

  def build(self, input_shape):
    init = tf.keras.initializers.GlorotNormal()
    bias_init = tf.keras.initializers.Zeros()
    self.w_prop = self.add_weight(name='w_prop',
                                  shape=(self.num_variables, self.num_filters),
                                  initializer=init,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
    if self.metric:
        self.w_metric = self.add_weight(name='w_metric',
                                       shape=(self.num_variables, self.num_filters),
                                       initializer=init,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        self.init_metric = self.add_weight(name="init_metric",
                                           shape=(1, self.num_variables),
                                           initializer=init,
                                           trainable=True)
    self.w_qual = self.add_weight(name='w_qual',
                                  shape=(self.num_filters,),
                                  initializer=init,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
    self.bias = self.add_weight(name='bias',
                                shape=(self.num_filters,),
                                initializer=bias_init,
                                trainable=True)
    self.init_run = self.add_weight(name="init_run",
                                       shape=(1, self.num_filters),
                                       initializer=init,
                                       trainable=True)
    super(LTLOperator, self).build(input_shape)

  def activation(self, x, training):
    if training:
        return tf.nn.sigmoid(self.slope*x)
    else:
        return tf.clip_by_value(tf.sign(x), 0, 1)

  def relu_scheduled(self, x, training):
    if training:
      return tf.maximum(self.alpha*x, x)
    else:
      return tf.nn.relu(x)

  @tf.function
  def call(self, traces, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    """
    Passes a batch of traces through the LTL Operator
    Input: (batch_size, trace_length, num_variables) tensor containing the traces
    Output: (batch_size, trace_length, num_filters) tensor containing the output
    """
    results = []

    curr_time_out = tf.tensordot(traces, self.w_prop, axes=((2), (0)))
    if self.metric:
        extension = tf.tile(tf.expand_dims(self.init_metric, 0), [traces.shape[0], 1, 1])
        traces_extended = tf.concat([traces, extension], axis=1)
        next_time_out = tf.tensordot(traces_extended, self.w_metric, axes=((2), (0)))
        next_time_out = next_time_out[:, 1:, :]
        combined_out = curr_time_out + next_time_out
    else:
        combined_out = curr_time_out
    # Loop through traces starting from last timestep (so we have access to run(i,k+1) at timestep k)
    next_run_val = self.activation(tf.tile(self.init_run, [traces.shape[0], 1]), training)
    for k in range(traces.shape[1]-1,-1,-1):
      # Grab all run summations for a single timestep
      curr_run_vals = tf.gather_nd(combined_out, list(zip(range(len(traces)), [k for _ in range(len(traces))])))
      partial_run = tf.math.multiply(self.relu_scheduled(self.w_qual, training), next_run_val)
      next_run_val = self.activation(partial_run + curr_run_vals + self.bias, training)
      results.append(next_run_val)
    # Flipping results lists since we iterate backwards
    results = tf.convert_to_tensor(list(reversed(results)))
    # Convert (trace_length, batch_size, num_filters) to (batch_size, trace_length, num_filters)
    results = tf.transpose(results, perm=[1, 0, 2])
    return results

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1], self.num_filters)
