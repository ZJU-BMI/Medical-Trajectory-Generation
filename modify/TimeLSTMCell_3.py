import numpy as np
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras import activations, initializers, regularizers, constraints
from tensorflow_core.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
import tensorflow as tf


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        inputs_x, inputs_t = inputs
        batch_size = array_ops.shape(inputs_x)[0]
        dtype = inputs_x.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state: '
            'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

    def create_zeros(unnested_state_size):
        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return array_ops.zeros(init_state_size, dtype=dtype)

    if nest.is_sequence(state_size):
        return nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)


# 添加一个门控单元时间信息：TIME_LSTM_1
class TimeLSTMCell_3(Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=False,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(TimeLSTMCell_3, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        if type(self.recurrent_initializer).__name__ == 'Identity':
            def recurrent_identity(shape, gain=1, dtype=None):
                del dtype
                return gain**np.concatenate(
                    [np.identity(shape[0])]*(shape[1] // shape[0]), axis=1)
            self.recurrent_initializer = recurrent_identity

        # w_h
        self.recurrent_kernel= self.add_weight(shape=(self.units, self.units*3),
                                               name='recurrent_kernel',
                                               initializer=self.recurrent_initializer,
                                               regularizer=self.recurrent_regularizer,
                                               constraint=self.recurrent_constraint)
        # w_x
        self.kernel = self.add_weight(shape=(input_shape[0][1], self.units * 5),
                                      name='kernel',
                                      initializer=self.recurrent_initializer,
                                      regularizer=self.recurrent_regularizer,
                                      constraint=self.recurrent_constraint)
        # w_t
        self.kernel_time = self.add_weight(shape=(1, self.units*3),
                                           name='kernel_time',
                                           initializer=self.recurrent_initializer,
                                           regularizer=self.recurrent_regularizer,
                                           constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                @K.eager_learning_phase_scope
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units*2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(shape=(self.units * 5,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None


        # w_x
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_c = self.kernel[:, self.units:self.units*2]
        self.kernel_o = self.kernel[:, self.units*2:self.units*3]
        self.kernel_t_1 = self.kernel[:, self.units*3:self.units*4]
        self.kernel_t_2 = self.kernel[:, self.units*4:]

        #w_t
        self.kernel_time_t_1 = self.kernel_time[:, :self.units]
        self.kernel_time_t_2 = self.kernel_time[:, self.units:self.units*2]
        self.kernel_time_o = self.kernel_time[:, self.units*2:]


        # w_h
        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernal_c = self.recurrent_kernel[:, self.units:self.units*2]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units*2:]

        if self.use_bias:
            self.bias_i = self.bias[:, :self.units]
            self.bias_c = self.bias[:, self.units:self.units*2]
            self.bias_o = self.bias[:, self.units*2:self.units*3]
            self.bias_t_1 = self.bias[:, self.units*3:self.units*4]
            self.bias_t_2 = self.bias[:, self.units*4:]
        else:
            self.bias_i = None
            self.bias_c = None
            self.bias_o = None
            self.bias_t_1 = None
            self.bias_t_2 = None

        self.build = True

    def call(self, inputs, states, training=None):
        assert isinstance(inputs, list)
        inputs_x, input_t = inputs

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs_x),
                self.dropout,
                training=training,
                count=3)

        if 0 < self.recurrent_dropout < 1 and self._dropout_mask is None:
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=3)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # h_(t-1)
        c_tm1 = states[1]  # c_(t-1)

        if 0 < self.dropout < 1:
            inputs_i = inputs_x * dp_mask[0]
            inputs_c = inputs_x * dp_mask[2]
            inputs_o = inputs_x * dp_mask[3]
            inputs_t = inputs_x * dp_mask[4]

        else:
            inputs_i = inputs_x
            inputs_c = inputs_x
            inputs_o = inputs_x
            inputs_t = inputs_x

        # x相关的所有数据
        x_i = K.dot(inputs_i, self.kernel_i)
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = K.dot(inputs_o, self.kernel_o)
        x_t_1 = K.dot(inputs_t, self.kernel_t_1)
        x_t_2 = K.dot(inputs_t, self.kernel_t_2)

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_c = K.bias_add(x_c, self.bias_c)
            x_o = K.bias_add(x_o, self.bias_o)
            x_t_1 = K.bias_add(x_t_1, self.bias_t_1)
            x_t_2 = K.bias_add(x_t_2, self.bias_t_2)

        if 0 < self.recurrent_dropout < 1:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_c = h_tm1 * rec_dp_mask[1]
            h_tm1_o = h_tm1 * rec_dp_mask[2]

        else:
            h_tm1_i = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        # 计算各个门控单元的过程
        i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))

        t_1 = self.recurrent_activation(x_t_1 + self.recurrent_activation(K.dot(input_t, self.kernel_time_t_1)))
        t_1_constraint = tf.where(tf.greater(t_1, tf.ones_like(t_1)*-0.00001), tf.ones_like(t_1)*-0.00001, t_1)

        t_2 = self.recurrent_activation(x_t_2 + self.recurrent_activation(K.dot(input_t, self.kernel_time_t_2)))

        c_m_ = (1-i*t_1)*c_tm1 + i*self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernal_c))*t_1_constraint

        c_m = (1-i)*c_tm1 + i*self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernal_c))*t_2

        o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o)+
                                      K.dot(input_t, self.kernel_time_o))

        h = self.activation(c_m_) * o
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._use_learning_phase = True
        return h, [h, c_m]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))