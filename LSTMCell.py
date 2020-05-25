import numpy as np
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras import activations, initializers
from tensorflow_core.python.keras.engine.base_layer import Layer


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


# 重新定义LSTMCell: 拆分成单个步骤计算
class LSTMCell(Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 **kwargs):
        super(LSTMCell, self).__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            **kwargs)

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint

        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.implementation = implementation

        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if type(self.recurrent_initializer).__name__ == 'Identity':
            def recurrent_identity(shape, gain=1, dtype=None):
                del dtype
                return gain**np.concatenate(
                    [np.identity(shape[0])]*(shape[1] // shape[0]), axis=1)
            self.recurrent_initializer = recurrent_identity

        self.kernel = self.add_weight(shape=(input_dim, self.units*4),
                                      name='kernel',
                                      initializer=self.recurrent_initializer,
                                      regularizer=self.recurrent_regularizer,
                                      constraint=self.recurrent_constraint)

        self.recurrent_kernel= self.add_weight(shape=(self.units, self.units*4),
                                               name='recurrent_kernel',
                                               initializer=self.recurrent_initializer,
                                               regularizer=self.recurrent_regularizer,
                                               constraint=self.recurrent_constraint)
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units*2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units*4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units:self.units*2]
        self.kernel_c = self.kernel[:, self.units*2, self.units*3]
        self.kernel_o = self.kernel[:, self.units*3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units:self.units*2]
        self.recurrent_kernal_c = self.recurrent_kernel[:, self.units*2:self.units*3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units*3:]

        if self.use_bias:
            self.bias_i = self.bias[:, :self.units]
            self.bias_f = self.bias[:, self.units:self.units*2]
            self.bias_c = self.bias[:, self.units*2, self.units*3]
            self.bias_o = self.bias[:, self.units*3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.build = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)

        if 0 < self.recurrent_dropout < 1 and self._dropout_mask is None:
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # h_(t-1)
        c_tm1 = states[1]  # c_(t-1)

        if self.implementation == 1:
            if 0 < self.dropout < 1:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]

            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)

            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]

            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1

            i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
            c = f*c_tm1 + i*self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernal_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))

        else:
            if 0 < self._dropout_mask < 1:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0 < self. recurrent_dropout < 1:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)

            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units, self.units*2]
            z2 = z[:, self.units*2, self.units*3]
            z3 = z[:, self.units*3:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f*c_tm1 + i*self.activation(z2)
            o = self.recurrent_activation(z3)

        h = self.activation(c)*o
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._use_learning_phase = True
        return h, [h, c]

