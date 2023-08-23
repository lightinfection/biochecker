import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from prepare_data import train_test_split, aug_data
import matplotlib.pyplot as plt
import numpy as np

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        if loss_type == 'epoch':
            # val_acc
            plt.subplot(1,2,1)
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc')
            plt.legend(loc="lower right")
            # val_loss
            plt.subplot(1,2,2)
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('loss')
            plt.legend(loc="upper right")
    
    def show(self):
        plt.show()
        
    def save(self, path, dpi):
        plt.savefig(fname=path, dpi=dpi)

class model():

    def __init__(self, train, validation, test):
        self._train_set = self.load_img(train)
        self._validation_set = self.load_img(validation)
        self._test_set = self.load_img(test)
        self.history = LossHistory()
        inputs, outputs = self._model_base(shp=self._train_set.shape[1:])
        self._model = keras.Model(inputs, outputs)
    
    def load_img(self, dataset):
        x = [tf.keras.utils.img_to_array(i[0]) for i in dataset]
        y = [i[1] for i in dataset]
        return (np.array(x), np.array(y))

    def _model_base(shp):
        w_i = keras.initializers.GlorotNormal(seed=1000)
        b_i = keras.initializers.Zeros()

        inputs = keras.Input(shape=(256, 340, 1))
        x = layers.Resizing(256,256)(inputs)
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(256,256,1), kernel_initializer=w_i, bias_initializer=b_i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

        x_skip = x
        x_skip = layers.Conv2D(filters=64, kernel_size=(1,1), strides=(1,1),kernel_initializer=w_i, bias_initializer=b_i)(x_skip)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", kernel_initializer=w_i, bias_initializer=b_i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, x_skip])
        x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

        x_skip = x
        x_skip = layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1),kernel_initializer=w_i, bias_initializer=b_i)(x_skip)
        x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", kernel_initializer=w_i, bias_initializer=b_i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, x_skip])
        x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

        x_skip = x
        x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", kernel_initializer=w_i, bias_initializer=b_i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, x_skip])
        x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        return inputs, outputs

    def _compile(self):
        self._model.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3,epsilon=1e-5),
                    metrics =['accuracy'])
    
    def _fit(self, batch_size, epochs):
        self._model.fit(self._train_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=self._validation_set,
                    callbacks=[self.history])
    
    def view_learning_curve(self):
        self.history.loss_plot('epoch')
        self.history.show()
    
    def save_learning_curve(self, path, dpi):
        self.history.loss_plot('epoch')
        self.history.save(path, dpi)

    def predict(self):
        cm =np.array(tf.math.confusion_matrix(predictions=
                                              np.array(tf.round(model.predict(self._test_set[0]))), 
                                              labels=self._test_set[1], 
                                              num_classes=2))
        self.acc = float((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))
        print("test_acc", self.acc)

def main():
    data = train_test_split(path_ctrl=os.getcwd()+"/A549 PCM image dataset/Original/Ctrl",path_cm=os.getcwd()+"/A549 PCM image dataset/Original/CM")
    data._split()
    classifier = model(train=data._train_set,validation=data._validation_set,test=data._test_set)
    print(classifier._model.summary())
    classifier._compile()
    try:
        classifier._fit()
        # classifier.view_learning_curve()
        # classifier.predict();
    except Exception as exception:
        print(exception)
        
if __name__ == "__main__":
    main()