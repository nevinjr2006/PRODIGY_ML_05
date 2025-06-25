import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load base model without top layer
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base.output)
output = Dense(101, activation='softmax')(x)
model = tf.keras.Model(inputs=base.input, outputs=output)

# Freeze layers initially
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators using Food-101 format
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
train_flow = train_gen.flow_from_directory('food-101/train', target_size=(224,224), batch_size=32)
valid_flow = train_gen.flow_from_directory('food-101/valid', target_size=(224,224), batch_size=32)

model.fit(train_flow, validation_data=valid_flow, epochs=10)
