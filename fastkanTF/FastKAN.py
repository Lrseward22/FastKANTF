import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import *

class RadialBasisFunction(layers.Layer):
    def __init__(
            self, 
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,
            **kw
        ):
        super(RadialBasisFunction, self).__init__(**kw)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = tf.linspace(grid_min, grid_max, num_grids)
        self.grid = tf.Variable(grid, trainable=False, name="grid")
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def call(self, inputs):
        return tf.exp(-((inputs[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(layers.Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_norm: bool = True,
        base_activation = tf.nn.silu,
        spline_weight_init_scale: float = 0.1,
        **kw
    ):
        super(FastKANLayer, self).__init__(**kw)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = layers.LayerNormalization(axis=-1) if (use_norm and input_dim > 1) else None
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.dense = layers.Dense(output_dim)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = layers.Dense(output_dim)

    def build(self, input_shape):
        spline_input = self.input_dim * self.rbf.num_grids
        self.dense.build((input_shape[0], spline_input))
        if self.use_base_update:
            self.base_linear.build((input_shape[0], self.input_dim))
        super().build(input_shape * self.rbf.num_grids)

    def call(self, inputs):
        norm_inputs = self.norm(inputs) if self.norm else inputs
        spline_basis = self.rbf(norm_inputs)
        out = self.dense(tf.reshape(spline_basis, [tf.shape(spline_basis)[0], -1]))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(inputs))
            out += base
        return out

class FastKAN(Model):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = tf.nn.silu,
        spline_weight_init_scale: float = 0.1,
        **kw
    ):
        super(FastKAN, self).__init__(**kw)
        self.shape = layers_hidden
        self.KAN_layers = [
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                use_norm=True,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ]

    def build(self, input_shape):
        for layer in self.KAN_layers:
            layer.build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.shape[-1])

    def call(self, inputs):
        for layer in self.KAN_layers:
            inputs = layer(inputs)
        return inputs

