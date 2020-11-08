# vgg16 model used for transfer learning on the dogs and cats dataset
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout


# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    add1 = Dropout(0.5)(class1)
    output = Dense(1, activation='sigmoid')(add1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')

    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plotfinal.png')
    pyplot.close()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')

    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot1final.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory('finalize_dogs_vs_cats/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    # fit model
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)
    mc = ModelCheckpoint('best_model.h5', monitor='acc', mode='max', verbose=1, save_best_only=True)
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, shuffle=True, callbacks=[es , mc])
    # save model
    model.save('final_model.h5')
    summarize_diagnostics(history)
# entry point, run the test harness
run_test_harness()
