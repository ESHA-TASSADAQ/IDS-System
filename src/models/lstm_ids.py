from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class LSTMIDSConfig:
	def __init__(
		self,
		input_timesteps: int,
		input_features: int,
		conv_filters: int = 64,
		conv_kernel_size: int = 3,
		pool_size: int = 2,
		lstm_units: int = 64,
		dropout_rate: float = 0.3,
		learning_rate: float = 1e-3,
	):
		self.input_timesteps = input_timesteps
		self.input_features = input_features
		self.conv_filters = conv_filters
		self.conv_kernel_size = conv_kernel_size
		self.pool_size = pool_size
		self.lstm_units = lstm_units
		self.dropout_rate = dropout_rate
		self.learning_rate = learning_rate


def build_lstm_ids_model(config: LSTMIDSConfig) -> tf.keras.Model:
	inputs = layers.Input(shape=(config.input_timesteps, config.input_features))
	# Conv1D + MaxPooling over the time dimension
	x = layers.Conv1D(filters=config.conv_filters, kernel_size=config.conv_kernel_size, activation="relu", padding="same")(inputs)
	x = layers.MaxPooling1D(pool_size=config.pool_size)(x)
	# LSTM
	x = layers.LSTM(config.lstm_units, return_sequences=False)(x)
	x = layers.Dropout(config.dropout_rate)(x)
	# Dense head
	x = layers.Dense(64, activation="relu")(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)

	model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_IDS")
	optimizer = optimizers.Adam(learning_rate=config.learning_rate)
	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
	return model


def count_trainable_params(model: tf.keras.Model) -> int:
	return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))