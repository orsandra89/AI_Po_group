import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
from configuration.env_variable import SAVE_DIR, TRAIN_DIR

# Create the save directory if it does not exist
os.makedirs(SAVE_DIR, exist_ok=True)

class Visualization:
    def __init__(self, model, history):
        self.model = model
        self.history = history

    def visualise_accuracy_and_loss(self):
        # Plot training and validation accuracy over epochs
        plt.figure()
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.savefig(os.path.join(SAVE_DIR, 'training_validation_accuracy.png'))
        plt.close()

        # Plot training and validation loss over epochs
        plt.figure()
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(SAVE_DIR, 'training_validation_loss.png'))
        plt.close()

    def visualise_dist_images_per_class(self):
        # Count the number of images in each class directory
        class_counts = {cls: len(os.listdir(os.path.join(TRAIN_DIR, cls))) for cls in os.listdir(TRAIN_DIR)}

        # Create a bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(class_counts.keys(), class_counts.values())

        # Explicitly set tick labels to ensure alignment
        plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=90)
        plt.xlabel("Plant Class")
        plt.ylabel("Number of Images")
        plt.title("Distribution of Images per Class in Training Dataset")
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(os.path.join(SAVE_DIR, 'distribution_of_images_per_class.png'))
        plt.close()