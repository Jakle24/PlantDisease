# Enhanced and Fixed TensorFlow Model Code for Leaf Disease Detection

from tensorflow.keras import Sequential, layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# point this path at the folder you unzipped from Kaggle
DATA_DIR = "../data/PlantVillage"

# instantiate generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

num_classes = len(train_data.class_indices)


# Assumes num_classes is defined
num_classes = 3  # e.g., 'healthy', 'early_blight', 'late_blight'

# 1. Data Preparation (make sure you define these properly in your main script)
train_data = ...  # your ImageDataGenerator or tf.data pipeline
val_data = ...
sample_image = ...  # a (1, 224, 224, 3) numpy array for feature extraction demo

# 2. Base MobileNetV2 Model
base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
base.trainable = False  # freeze for transfer learning


# 3. Transfer Learning Model Definition
model = Sequential([
    base,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# 4. Compile and Train (Phase 1)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_data, epochs=5, validation_data=val_data)

# 5. Fine-tuning Phase (Unfreeze last few layers of MobileNetV2)
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_data, epochs=5, validation_data=val_data)

# 6. Feature Extraction Model (Optional)
# Get outputs from base model's convolutional block for visualisation/debugging
feature_model = Model(
    inputs=model.input,
    outputs=[base.get_layer(index=i).output for i in range(10, 13)]  # adjust indices as needed
)

features = feature_model.predict(sample_image)

# 7. Save Final Model
model.save('plant_disease_model.h5')
