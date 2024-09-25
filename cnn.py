#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model, Sequential, load_model
from keras.saving import register_keras_serializable
#
from keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input, Dropout, Flatten, Permute, Reshape, Lambda, concatenate
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.layers import Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
#
from keras_nlp.layers import PositionEmbedding, TransformerEncoder, TransformerDecoder
#
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
#
from tensorflow.keras.utils import Progbar
#
from random import randint, random, shuffle
from math import log, tanh
from os import system
import concurrent.futures
from tqdm import tqdm
import time
import struct as st

@register_keras_serializable()
def t2d(x): return tf.transpose(x, perm=[0, 2, 1])
@register_keras_serializable()
def tos(input_shape): return (input_shape[0], input_shape[2], input_shape[1])

@register_keras_serializable()
class GaussianActivation(Layer):
	def call(self, x): return tf.exp(-tf.square(x) * 5)

def mul(l):
	a = 1
	for e in l: a*=e
	return a

tf_logistique = lambda x: 1/(1+tf.exp(-         x ))
np_logistique = lambda x: 1/(1+np.exp(-np.array(x)))
logistique    = lambda x: 1/(1+np.exp(-         x ))

@register_keras_serializable()
def custom_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	#
	w   = y_true[:, 0:1]
	#yh = y_true[:, 1:2]
	#
	y = tf.tanh      (y0)
	h = tf_logistique(y1)
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	Y = tf.pow(tf.sign(w) - y0, 2)/2
	#
	return (tf.reduce_mean(Y))# + tf.reduce_mean(H))/2

#	============================================================	#

from cree_les_données import df, VALIDATION, N, nb_expertises, T, DEPART, SORTIES

X_train = (T-DEPART-VALIDATION, N, nb_expertises)
Y_train = (T-DEPART-VALIDATION, 1)
X_test  = (         VALIDATION, N, nb_expertises)
Y_test  = (         VALIDATION, 1)

for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
	with open(la_liste, 'rb') as co:
		bins = co.read()
		exec(f"{la_liste} = np.array(st.unpack('f'*mul({la_liste}), bins)).reshape({la_liste})")

#	============================================================	#

if __name__ == "__main__":
	entree = Input((N, nb_expertises))#Input((nb_expertises, N))
	x = entree
	#
	#Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose,
	x = Conv1D(64, 3)(x)	#8 -> 6
	x = Conv1D(32, 3)(x)	#6 -> 4
	x = Conv1D(16, 3)(x)	#4 -> 2
	#
	x = Flatten()(x)
	x = Dropout(0.10)(x)
	#
	x = Dense(128, activation='relu')(x); x = Dropout(0.50)(x)
	#
	x = Dense(U, activation='softmax')(x)

	model = Model(entree, x)
	model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy')#custom_loss)
	model.summary()

	############################ Entrainnement #########################

	# Callbacks
	meilleur_validation = ModelCheckpoint('meilleur_model.h5.keras', monitor='val_loss', save_best_only=True)
	meilleur_train      = ModelCheckpoint('dernier__model.h5.keras', monitor='loss'    , save_best_only=True)
	
	history = model.fit(X_train, Y_train, epochs=200, batch_size=256, validation_data=(X_test,Y_test), shuffle=True,
		callbacks=[
			meilleur_validation, meilleur_train
		]
	)

	plt.plot(history.history['loss'    ], label='Train')
	plt.plot(history.history['val_loss'], label='Test ')
	plt.legend()
	plt.show()

	print("======================================================================")

	def display_layer_outputs(model, input_data):
		# Créer un modèle qui retourne les sorties de chaque couche
		layer_outputs = [layer.output for layer in model.layers]
		activation_model = Model(inputs=model.input, outputs=layer_outputs)

		# Obtenir les valeurs de chaque couche
		activations = activation_model.predict(input_data)

		# Afficher les valeurs de chaque couche
		for layer_name, activation in zip([layer.name for layer in model.layers], activations):
			print(f"Layer: {layer_name}, Activation shape: {activation.shape}, Activation values:\n{activation}")

	display_layer_outputs(model, X_train[0:1])

	print("======================================================================")

	for layer in model.layers:
		if 'conv' in layer.name:
			# Obtenir les poids de la couche (les kernels sont dans layer.get_weights()[0])
			kernels = layer.get_weights()[0]
			
			# Normaliser les valeurs des kernels pour les afficher correctement
			min_val = np.min(kernels)
			max_val = np.max(kernels)
			kernels = (kernels - min_val) / (max_val - min_val)

			# Déterminer le nombre de filtres et la taille des kernels
			num_filters = kernels.shape[-1]
			kernel_size = kernels.shape[0]

			# Créer un plot pour afficher les filtres
			A = int(1 + num_filters**.5)
			fig, ax = plt.subplots(A, A)
			
			for i in range(num_filters):
				for ie in range(nb_expertises):
					ax[i//A][i%A].plot(kernels[:, ie, i])
			plt.show()