Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


input_shape = (299, 299, 3)
num_classes = 15  
batch_size = 32
epochs = 50
learning_rate = 0.0001


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)


train_dir = 'path_to_training_data'
test_dir = 'path_to_testing_data'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)
... 
... 
... base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
... x = base_model.output
... x = GlobalAveragePooling2D()(x)
... x = Dense(512, activation='relu')(x)
... x = Dropout(0.5)(x)  
... predictions = Dense(num_classes, activation='softmax')(x)
... 
... model = Model(inputs=base_model.input, outputs=predictions)
... 
... 
... for layer in base_model.layers:
...     layer.trainable = False
... 
... 
... model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
...               loss='categorical_crossentropy',
...               metrics=['accuracy'])
... 
... 
... checkpoint = ModelCheckpoint('inceptionv3_best_model.h5', monitor='val_accuracy', save_best_only=True)
... early_stopping = EarlyStopping(monitor='val_loss', patience=5)
... 
... 
... history = model.fit(
...     train_generator,
...     steps_per_epoch=train_generator.samples // batch_size,
...     validation_data=test_generator,
...     validation_steps=test_generator.samples // batch_size,
...     epochs=epochs,
...     callbacks=[checkpoint, early_stopping]
... )
... 
... 
... test_loss, test_accuracy = model.evaluate(test_generator)
... print(f"Test Accuracy: {test_accuracy:.2f}")
