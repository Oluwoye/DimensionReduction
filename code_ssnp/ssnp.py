import gc
import os
import numpy as np
from tensorflow.keras import initializers
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import losses
from tensorflow.keras import datasets as kdatasets
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class SSNP():
    def __init__(self, init_labels='precomputed', epochs=100, input_l1=0.0, input_l2=0.0, bottleneck_l1=0.0,
                 bottleneck_l2=0.5, verbose=1, opt='adam', bottleneck_activation='linear', act='relu',
                 init='glorot_uniform', bias=0.0001, patience=3, min_delta=0.01, mode=0):
        self.init_labels = init_labels
        self.epochs = epochs
        self.verbose = verbose
        self.opt = opt
        self.act = act
        if init == "glorot_uniform":
            self.init = initializers.GlorotUniform(seed=420)
        elif init == "random_normal":
            self.init = initializers.RandomNormal(seed=420)
        elif init == "randorm_uniform":
            self.init = initializers.RandomUniform(seed=420)
        elif init == "truncated_normal":
            self.init = initializers.TruncatedNormal(seed=420)
        elif init == "zeros":
            self.init = initializers.Zeros()
        elif init == "ones":
            self.init = initializers.Ones()
        elif init == "glorot_normal":
            self.init = initializers.GlorotNormal(seed=420)
        elif init == "he_normal":
            self.init = initializers.HeNormal(seed=420)
        elif init == "he_uniform":
            self.init = initializers.HeUniform(seed=420)
        elif init == "identity":
            self.init = initializers.Identity()
        elif init == "orthogonal":
            self.init = initializers.Orthogonal(seed=420)
        else:
            ValueError("Unrecognized model layer initializer")
        self.bias = bias
        self.input_l1 = input_l1
        self.input_l2 = input_l2
        self.bottleneck_l1 = bottleneck_l1
        self.bottleneck_l2 = bottleneck_l2
        self.bottleneck_activation = bottleneck_activation
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.label_bin = LabelBinarizer()

        self.fwd = None
        self.inv = None

        tf.random.set_seed(420)

        self.is_fitted = False
        K.clear_session()

    def fit(self, X, y=None):
        if y is None and self.init_labels == 'precomputed':
            raise Exception('Must provide labels when using init_labels = precomputed')

        if y is None:
            y = self.init_labels.fit_predict(X)

        n_classes = len(np.unique(y))
        if n_classes == 2:
            y = np.array([[1, 0] if el == y[0] else [0, 1] for el in y])
        else:
            self.label_bin.fit(y)

        decoder_output, encoded, main_input, main_output = self.get_ae_model(X, n_classes, model_mode=self.mode)

        model = Model(inputs=main_input, outputs=[main_output, decoder_output])

        model.compile(optimizer=self.opt,
                      loss={'main_output': 'categorical_crossentropy', 'decoder_output': 'binary_crossentropy'},
                      metrics=['accuracy'])

        if self.patience > 0:
            callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=self.min_delta, patience=self.patience,
                                       restore_best_weights=True, verbose=self.verbose)]
        else:
            callbacks = []

        if n_classes == 2:
            hist = model.fit(X,
                             [y, X],
                             batch_size=32,
                             epochs=self.epochs,
                             shuffle=True,
                             verbose=self.verbose,
                             validation_split=0.05,
                             callbacks=callbacks)
        else:
            hist = model.fit(X,
                             [self.label_bin.transform(y), X],
                             batch_size=32,
                             epochs=self.epochs,
                             shuffle=True,
                             verbose=self.verbose,
                             validation_split=0.05,
                             callbacks=callbacks)

        encoded_input = Input(shape=(2,))
        l = model.get_layer('enc1')(encoded_input)
        l = model.get_layer('enc2')(l)
        l = model.get_layer('enc3')(l)
        decoder_layer = model.get_layer('decoder_output')(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded)
        self.clustering = Model(inputs=main_input, outputs=main_output)
        self.is_fitted = True

        return hist

    def get_ae_model(self, X, n_classes, model_mode=0):
        main_input = Input(shape=(X.shape[1],), name='main_input')
        if model_mode == 0:
            x = Dense(512, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(128, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(32, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(32, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(128, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(512, activation=self.act, kernel_initializer=self.init, name='enc3',
                      bias_initializer=Constant(self.bias))(x)
        elif model_mode == 1:
            x = Dense(256, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(64, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(16, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(16, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(64, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(256, activation=self.act, kernel_initializer=self.init, name='enc3',
                      bias_initializer=Constant(self.bias))(x)
        elif model_mode == 2:
            x = Dense(64, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(32, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(8, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(8, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(32, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(64, activation=self.act, kernel_initializer=self.init, name='enc3',
                      bias_initializer=Constant(self.bias))(x)
        elif model_mode == 3:
            x = Dense(128, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(32, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(32, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(128, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
        elif model_mode == 4:
            x = Dense(64, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(16, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(16, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(64, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
        elif model_mode == 5:
            x = Dense(1024, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(512, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(64, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(64, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(512, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(1024, activation=self.act, kernel_initializer=self.init, name='enc3',
                      bias_initializer=Constant(self.bias))(x)
        elif model_mode == 6:
            x = Dense(1024, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(main_input)
            x = Dense(512, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(128, activation=self.act,
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(32, activation=self.act,
                      activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                      kernel_initializer=self.init,
                      bias_initializer=Constant(self.bias))(x)
            encoded = Dense(2,
                            activation=self.bottleneck_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
            x = Dense(32, activation=self.act, kernel_initializer=self.init, name='enc1',
                      bias_initializer=Constant(self.bias))(encoded)
            x = Dense(128, activation=self.act, kernel_initializer=self.init, name='enc2',
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(512, activation=self.act, kernel_initializer=self.init, name='enc3',
                      bias_initializer=Constant(self.bias))(x)
            x = Dense(1024, activation=self.act, kernel_initializer=self.init, name='enc4',
                      bias_initializer=Constant(self.bias))(x)
        if n_classes == 2:
            n_units = 2
        else:
            n_units = n_classes
        main_output = Dense(n_units,
                            activation='softmax',
                            name='main_output',
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)
        decoder_output = Dense(X.shape[1],
                               activation='sigmoid',
                               name='decoder_output',
                               kernel_initializer=self.init,
                               bias_initializer=Constant(self.bias))(x)
        return decoder_output, encoded, main_input, main_output

    def transform(self, X):
        if self._is_fit():
            return self.fwd.predict(X)

    def inverse_transform(self, X_2d):
        if self._is_fit():
            return self.inv.predict(X_2d)

    def predict(self, X):
        if self._is_fit():
            y_pred = self.clustering.predict(X)
            return self.label_bin.inverse_transform(y_pred)

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception('Model not trained. Call fit() before calling transform()')
