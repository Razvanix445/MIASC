import os
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
import seaborn as sns

import tensorflow as tf

print(tf.__version__)
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "data/preprocessed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def load_preprocessed_data(data_dir):
    """Load preprocessed data from the specified directory"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    class_indices = np.load(os.path.join(data_dir, 'class_indices.npy'), allow_pickle=True).item()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of classes: {len(class_indices)}")

    return X_train, y_train, X_test, y_test, class_indices


X_train, y_train, X_test, y_test, class_indices = load_preprocessed_data(DATA_DIR)

y_train_indices = np.array([class_indices[label] for label in y_train])
y_test_indices = np.array([class_indices[label] for label in y_test])

num_classes = len(class_indices)
y_train_cat = to_categorical(y_train_indices, num_classes)
y_test_cat = to_categorical(y_test_indices, num_classes)

index_to_class = {v: k for k, v in class_indices.items()}

class_counts = {}
for class_name in class_indices.keys():
    train_count = np.sum(y_train == class_name)
    test_count = np.sum(y_test == class_name)
    class_counts[class_name] = (train_count, test_count)

print("\nClass distribution (train, test):")
for class_name, (train_count, test_count) in class_counts.items():
    print(f"{class_name}: {train_count} train, {test_count} test")


def plot_training_history(history, model_name):
    """Plot training and validation accuracy and loss"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'{model_name}_training_history.png'))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'{model_name}_confusion_matrix.png'))
    plt.show()


def evaluate_model(model, X_test, y_test_cat, y_test_indices, class_names, model_name):
    """Evaluate model and return metrics"""
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

    y_pred_probs = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    precision = precision_score(y_test_indices, y_pred_indices, average='weighted')
    recall = recall_score(y_test_indices, y_pred_indices, average='weighted')
    f1 = f1_score(y_test_indices, y_pred_indices, average='weighted')

    print(f"\n{model_name} Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    report = classification_report(y_test_indices, y_pred_indices, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    plot_confusion_matrix(y_test_indices, y_pred_indices, class_names, model_name)

    metrics = {
        'model_name': model_name,
        'accuracy': test_accuracy,
        'loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics


def visualize_predictions(model, X_test, y_test, index_to_class, model_name, num_samples=5):
    """Visualize model predictions on test samples"""
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)

    predictions = model.predict(X_test[indices])
    predicted_classes = [index_to_class[np.argmax(pred)] for pred in predictions]
    true_classes = [y_test[i] for i in indices]

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test[idx])
        plt.title(f"True: {true_classes[i]}\nPred: {predicted_classes[i]}")
        plt.axis('off')

    plt.suptitle(f"{model_name} - Sample Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'{model_name}_sample_predictions.png'))
    plt.show()


def build_baseline_cnn(input_shape, num_classes):
    """Build a simple baseline CNN model"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(16, (3, 3), activation='linear', padding='same', input_shape=input_shape),
        MaxPooling2D((4, 4)),

        # Second Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Flatten and Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.8),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_baseline_cnn(X_train.shape[1:], num_classes)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

print("Training Baseline CNN Model...")
history = model.fit(
    X_train, y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

plot_training_history(history, 'Baseline_CNN')

metrics = evaluate_model(
    model, X_test, y_test_cat, y_test_indices,
    list(class_indices.keys()), 'Baseline_CNN'
)

visualize_predictions(model, X_test, y_test, index_to_class, 'Baseline_CNN')

model.save(os.path.join(MODEL_DIR, 'baseline_cnn.h5'))

df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)


def predict_disease(model, img_path, class_indices):
    """Make prediction for a single image"""
    img = load_and_preprocess_image(img_path)

    pred = model.predict(img)
    pred_class_idx = np.argmax(pred[0])
    pred_prob = pred[0][pred_class_idx]

    index_to_class = {v: k for k, v in class_indices.items()}
    pred_class = index_to_class[pred_class_idx]

    return pred_class, pred_prob, pred[0]


def sample_inference(model_path, test_img_paths, class_indices):
    """Run sample inference and visualize results"""
    model = load_model(model_path)

    plt.figure(figsize=(15, 4 * len(test_img_paths)))
    for i, img_path in enumerate(test_img_paths):
        pred_class, pred_prob, all_probs = predict_disease(model, img_path, class_indices)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(len(test_img_paths), 2, i * 2 + 1)
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class}\nConfidence: {pred_prob:.2%}")
        plt.axis('off')

        plt.subplot(len(test_img_paths), 2, i * 2 + 2)
        index_to_class = {v: k for k, v in class_indices.items()}
        class_names = [index_to_class[i] for i in range(len(class_indices))]
        colors = ['green' if class_name == pred_class else 'blue' for class_name in class_names]
        plt.bar(class_names, all_probs, color=colors)
        plt.title('Class Probabilities')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'sample_inference.png'))
    plt.show()


import random

test_img_paths = []
for cls in class_indices.keys():
    test_dir = os.path.join(DATA_DIR, 'test', cls)
    if os.path.exists(test_dir):
        img_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if img_files:
            test_img_paths.append(os.path.join(test_dir, random.choice(img_files)))

model_path = os.path.join(MODEL_DIR, 'baseline_cnn.h5')
if len(test_img_paths) > 0:
    sample_inference(model_path, test_img_paths[:3], class_indices)