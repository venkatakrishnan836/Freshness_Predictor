#CODE TO TRAIN THE MODEL

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {gpus}")
else:
    print("No GPU found, using CPU.")

# Configure TensorFlow to use GPU if available
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(f"Error setting up GPU: {e}")

# Directories for the training dataset
train_dir = "Path to Train Dataset"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.96
)

base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model_resnet.layers[:-10]:  
    layer.trainable = False
for layer in base_model_resnet.layers[-10:]:
    layer.trainable = True

model_resnet = Sequential([
    base_model_resnet,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)), 
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  
])

optimizer = Adam(learning_rate=lr_schedule)
model_resnet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model_resnet.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history_resnet = model_resnet.fit(
    train_generator,
    epochs=150,  # Increas number of epochs
    validation_data=validation_generator,
    callbacks=[early_stopping]
)




#CODE TO TEST THE MODEL

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Load the trained ResNet50 model
model_resnet = load_model('resnet50_fruit_freshness_classifier.h5')
print("Model loaded successfully.")

class_labels = ['freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 'freshoranges', 
                'freshpotato', 'freshtomato', 'rottenapples', 'rottenbanana', 'rottencucumber', 
                'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato']

def predict_and_analyze_images_from_folder(folder_path, model, class_labels, true_labels=None):
    y_true = [] 
    y_pred = []  
    confidences = []  

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            print(f"Skipping non-image file: {img_path}")
            continue

        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_preprocessed = np.expand_dims(img_array / 255.0, axis=0) 

        # Make prediction
        pred = model.predict(img_preprocessed)
        confidence = np.max(pred)  # Get confidence score
        predicted_class_index = np.argmax(pred)
        predicted_class = class_labels[predicted_class_index]

        true_class = true_labels[y_pred.index(predicted_class_index)] if true_labels else None
        
        y_pred.append(predicted_class_index)
        confidences.append(confidence)
        if true_labels:
            y_true.append(class_labels.index(true_class))

        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%")
        plt.axis('off')
        plt.show()

        print(f"Image: {img_path}")
        print(f"Predicted Class: {predicted_class} (Confidence: {confidence*100:.2f}%)")
        if true_labels:
            print(f"True Class: {true_class}")
        print()

    if true_labels:
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

# Specify the folder containing the images
folder_path = r"path to test image folder"

true_labels = [
    'freshapples', 'freshapples', 'rottenapples', 'rottenapples',
    'freshbanana', 'freshbanana', 'rottenbanana', 'rottenbanana',
    'freshcucumber', 'freshcucumber', 'rottencucumber', 'rottencucumber',
    'freshokra', 'freshokra', 'rottenokra', 'rottenokra',
    'freshoranges', 'freshoranges', 'rottenoranges', 'rottenoranges',
    'freshpotato', 'freshpotato', 'rottenpotato', 'rottenpotato',
    'freshtomato', 'freshtomato', 'rottentomato', 'rottentomato'
]

predict_and_analyze_images_from_folder(folder_path, model_resnet, class_labels, true_labels)


model_resnet.save('resnet50_fruit_freshness_classifier2.h5')
print("Model saved successfully.")
