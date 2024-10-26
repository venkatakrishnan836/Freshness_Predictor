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

# Class labels (adjust according to your dataset)
class_labels = ['freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 'freshoranges', 
                'freshpotato', 'freshtomato', 'rottenapples', 'rottenbanana', 'rottencucumber', 
                'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato']

# Function to predict, plot, and analyze results for images in a folder
def predict_and_analyze_images_from_folder(folder_path, model, class_labels, true_labels=None):
    y_true = []  # Store true labels (optional if available)
    y_pred = []  # Store predicted labels
    confidences = []  # Store confidence scores

    # Iterate over each file in the specified folder
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image (you can add more formats as needed)
        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            print(f"Skipping non-image file: {img_path}")
            continue

        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_preprocessed = np.expand_dims(img_array / 255.0, axis=0)  # Normalize the image

        # Make prediction
        pred = model.predict(img_preprocessed)
        confidence = np.max(pred)  # Get confidence score
        predicted_class_index = np.argmax(pred)
        predicted_class = class_labels[predicted_class_index]

        # Optionally get the true label if provided
        true_class = true_labels[y_pred.index(predicted_class_index)] if true_labels else None
        
        # Store predictions for confusion matrix and classification report
        y_pred.append(predicted_class_index)
        confidences.append(confidence)
        if true_labels:
            y_true.append(class_labels.index(true_class))

        # Display the image with predicted class and confidence score
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%")
        plt.axis('off')
        plt.show()

        # Print the prediction information
        print(f"Image: {img_path}")
        print(f"Predicted Class: {predicted_class} (Confidence: {confidence*100:.2f}%)")
        if true_labels:
            print(f"True Class: {true_class}")
        print()

    # If true labels are provided, generate a classification report and confusion matrix
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
# Example true_labels list (ensure this matches the order of image files in the folder)
true_labels = [
    'freshapples', 'freshapples', 'rottenapples', 'rottenapples',
    'freshbanana', 'freshbanana', 'rottenbanana', 'rottenbanana',
    'freshcucumber', 'freshcucumber', 'rottencucumber', 'rottencucumber',
    'freshokra', 'freshokra', 'rottenokra', 'rottenokra',
    'freshoranges', 'freshoranges', 'rottenoranges', 'rottenoranges',
    'freshpotato', 'freshpotato', 'rottenpotato', 'rottenpotato',
    'freshtomato', 'freshtomato', 'rottentomato', 'rottentomato'
]

# Call the function to predict and analyze images from the folder
predict_and_analyze_images_from_folder(folder_path, model_resnet, class_labels, true_labels)
