#! /usr/bin/python3

from from_import import *

###################################################

@register_keras_serializable()
class DiscretizationLayer(Layer):
	def __init__(self, thresholds, **kwargs):
		super(DiscretizationLayer, self).__init__(**kwargs)
		self.thresholds = thresholds

	def call(self, inputs):
		# Utiliser tf.digitize pour discrétiser selon les seuils
		discretized = tf.raw_ops.Bucketize(input=inputs, boundaries=self.thresholds)
		return tf.cast(discretized, tf.float32)
		
@register_keras_serializable()
class AffineTransformPerVector(Layer):
	def __init__(self, **kwargs):
		super(AffineTransformPerVector, self).__init__(**kwargs)

	def build(self, input_shape):
		S =  (*input_shape[1:-1], 1,)
		self.alpha = self.add_weight(shape=S,
									 initializer='ones',
									 trainable=True,
									 name='alpha')
		self.beta = self.add_weight(shape=S,
									initializer='zeros',
									trainable=True,
									name='beta')

	def call(self, x):
		return self.alpha * x + self.beta

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

from cree_les_données import df, VALIDATION, N, nb_expertises, T, DEPART, SORTIES, CLASSES, U

X_train = (T-DEPART-VALIDATION, N, nb_expertises)
Y_train = (T-DEPART-VALIDATION, CLASSES)
X_test  = (         VALIDATION, N, nb_expertises)
Y_test  = (         VALIDATION, CLASSES)

for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
	with open(la_liste, 'rb') as co:
		bins = co.read()
		exec(f"{la_liste} = np.array(st.unpack('f'*mul({la_liste}), bins)).reshape({la_liste})")

#	============================================================	#

def couche_convolution(f, k=3, activation='gelu'):
	return Sequential([
		Conv1D(f, k, activation=activation),
		BatchNormalization(),
		AveragePooling1D(2),
		Dropout(0.30)
	])

if __name__ == "__main__":
	entree = Input((N, nb_expertises))
	x = entree
	#
	#Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose,
	#
	x = couche_convolution(64, 5, 'linear')(x)	#8 -> 4 -> 2
	#x = couche_convolution(32, 3, 'linear')(x)	#6 -> 4
	#x = couche_convolution(64, 3, 'linear')(x)	#4 -> 2
	#
	#-----------------#
	x = Flatten()(x)  #
	#-----------------#
	#
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.30)(x)
	#
	x = Dense(SORTIES, activation='softmax')(x)

	model = Model(entree, x)
	#model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy')#custom_loss)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()

	############################ Entrainnement #########################

	# Callbacks
	meilleur_validation = ModelCheckpoint('meilleur_model.h5.keras', monitor='val_loss', save_best_only=True)
	meilleur_train      = ModelCheckpoint('dernier__model.h5.keras', monitor='loss'    , save_best_only=True)
	
	history = model.fit(X_train, Y_train, epochs=150, batch_size=256, validation_data=(X_test,Y_test), shuffle=True,
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