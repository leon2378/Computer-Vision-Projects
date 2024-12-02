import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore

#from sklearn.metrics import confusion_matrix, classification_report

# Define parameters
batch_size = 128
img_height = 64
img_width = 64

# create an image generator for train
#train_datagen = ImageDataGenerator(rescale=1./255)

#valid_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator(rescale=1./255)

# load the train data
train_dataset = image_dataset_from_directory(
    'train',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

# get class names from the train dataset
class_names = train_dataset.class_names
print(f'Class names are: [{class_names}]')

# load valid data
valid_dataset = image_dataset_from_directory(
    'valid',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# load test data
test_dataset = image_dataset_from_directory(
    'test',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# data augmentation
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal')
    #tf.keras.layers.RandomRotation(0.2)
    #tf.keras.layers.RandomZoom(0.1),             # Randomly zooms images by 10%
    #tf.keras.layers.RandomContrast(0.2),         # Randomly adjusts contrast by 20%
    #tf.keras.layers.RandomBrightness(0.2),
    #tf.keras.layers.RandomCrop(img_height, img_width), # Randomly crops image to target size
    #tf.keras.layers.RandomTranslation(0.1, 0.1) # Add random translation
])

# normalize the images
norm_layer = tf.keras.layers.Rescaling(1./255)

def preprocess(image, label):
    image = data_aug(image)
    image = norm_layer(image)
    return image, label

train_dataset = train_dataset.map(preprocess)
valid_dataset = valid_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

base_model = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

# freeze the base model
base_model.trainable = True

# add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(4096, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.01))(x)
x = Dropout(0.5)(x)

predictions = Dense(len(class_names), activation = 'softmax')(x)

# create the final model
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# train the model
epochs = 80


history = model.fit(train_dataset, 
                    epochs=epochs, 
                    validation_data=valid_dataset)

'''
# unfreeze some layers for fine tuning
base_model.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

fine_tune_epochs = 50
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(train_dataset, 
                         epochs=total_epochs, 
                         initial_epoch=history.epoch[-1], 
                         validation_data=valid_dataset)
'''
# evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f'test accuracy: {accuracy}')
