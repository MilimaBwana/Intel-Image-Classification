from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import keras.applications
from keras.callbacks import TensorBoard
import util_functions
from keras import backend as K

'Needed if executed as jupyter notebook on colab with data on google drive'
#drive.mount('/content/drive')

img_shape = (150, 150, 3)
tb_log_dir = 'logs'
num_epochs = 2
num_classes = 6


def build_model():
    basemodel = keras.applications.vgg16.VGG16(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg')

    for layer in basemodel.layers:
        layer.trainable = False

    basemodel.summary()

    model = Sequential()
    model.add(basemodel)
    '''
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))'''
    model.add(Dense(6, activation='softmax'))

    return model

# TODO: doesnt seem to be correct
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def main():
    seed = 42

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        data_format='channels_last',
        validation_split=0.25,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        'intel-image-classification/seg_train',
        target_size=img_shape[:-1], # only height and width
        batch_size=32,
        seed=seed,
        shuffle=True,
        color_mode='rgb',
        class_mode='sparse')

    test_data = test_datagen.flow_from_directory(
        'intel-image-classification/seg_test',
        target_size=img_shape[:-1], # only height and width
        color_mode='rgb',
        class_mode='sparse')

    util_functions.clear_folder(tb_log_dir)
    tensorboard = TensorBoard(log_dir=tb_log_dir)

    model = build_model()
    optimizer = optimizers.RMSprop(lr=1e-4)
    # Labels are integers
    print('Start Training')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc', precision_m, recall_m])
    print(model.summary)
    model.fit_generator(
        train_data,
        epochs=num_epochs,
        callbacks=[tensorboard]
    )

    model.evaluate_generator(
        test_data
    )


if __name__ == "__main__":
    main()
