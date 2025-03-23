import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import seaborn as sns
from pathlib import Path
import joblib
import datetime

# Deep learning libraries
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model

# Evaluation libraries
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set up directories
DATA_DIR = "data/preprocessed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if GPU is available
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def load_preprocessed_data(data_dir):
    """Load preprocessed data from the specified directory"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Load class indices
    class_indices = np.load(os.path.join(data_dir, 'class_indices.npy'), allow_pickle=True).item()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of classes: {len(class_indices)}")

    return X_train, y_train, X_test, y_test, class_indices


# Load data
X_train, y_train, X_test, y_test, class_indices = load_preprocessed_data(DATA_DIR)

# Convert class labels to indices
y_train_indices = np.array([class_indices[label] for label in y_train])
y_test_indices = np.array([class_indices[label] for label in y_test])

# Convert to one-hot encoding
num_classes = len(class_indices)
y_train_cat = to_categorical(y_train_indices, num_classes)
y_test_cat = to_categorical(y_test_indices, num_classes)

# Reverse mapping (index to class name)
index_to_class = {v: k for k, v in class_indices.items()}

# Print class distribution
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

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
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
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    precision = precision_score(y_test_indices, y_pred_indices, average='weighted')
    recall = recall_score(y_test_indices, y_pred_indices, average='weighted')
    f1 = f1_score(y_test_indices, y_pred_indices, average='weighted')

    # Print evaluation metrics
    print(f"\n{model_name} Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print classification report
    report = classification_report(y_test_indices, y_pred_indices, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    # Plot confusion matrix
    plot_confusion_matrix(y_test_indices, y_pred_indices, class_names, model_name)

    # Return metrics dictionary
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
    # Get random sample indices
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)

    # Make predictions
    predictions = model.predict(X_test[indices])
    predicted_classes = [index_to_class[np.argmax(pred)] for pred in predictions]
    true_classes = [y_test[i] for i in indices]

    # Plot
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
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # Flatten and Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Build Model 1
model1 = build_baseline_cnn(X_train.shape[1:], num_classes)
model1.summary()


def build_advanced_cnn(input_shape, num_classes):
    """Build a more advanced CNN with BatchNormalization"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Build Model 2
model2 = build_advanced_cnn(X_train.shape[1:], num_classes)
model2.summary()


def build_transfer_learning_model(input_shape, num_classes):
    """Build a transfer learning model using MobileNetV2"""
    # Load pre-trained MobileNetV2 model without top layers
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model on top
    inputs = Input(shape=input_shape)
    x = inputs

    # Ensure input is properly scaled for MobileNetV2 (expects [-1, 1])
    # Our images are already normalized to [0, 1], so we scale to [-1, 1]
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(x)

    # Pass the input through the base model
    x = base_model(x, training=False)

    # Add custom top layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Build Model 3
model3 = build_transfer_learning_model(X_train.shape[1:], num_classes)
model3.summary()


# Define data augmentation for training
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Visualize some augmented images
# Visualize some augmented images
def visualize_augmentation(X_train, y_train, augmentation, num_samples=5):
    """Visualize augmented images"""
    indices = np.random.choice(range(len(X_train)), num_samples, replace=False)
    X_samples = X_train[indices]

    plt.figure(figsize=(15, 4 * num_samples))
    for i, img_idx in enumerate(range(num_samples)):
        original_img = X_samples[img_idx]
        aug_img_iter = augmentation.flow(np.expand_dims(original_img, 0), batch_size=1)

        # Original image
        plt.subplot(num_samples, 5, i*5 + 1)
        plt.imshow(original_img)
        plt.title("Original")
        plt.axis('off')

        # 4 Augmented versions
        for j in range(4):
            # Change this line - use next() function instead of .next() method
            aug_img = next(aug_img_iter)[0]
            plt.subplot(num_samples, 5, i*5 + j + 2)
            plt.imshow(aug_img)
            plt.title(f"Augmented {j+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'data_augmentation_examples.png'))
    plt.show()

# Visualize data augmentation
visualize_augmentation(X_train, y_train, data_augmentation)


# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model1_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train Model 1
print("Training Model 1: Baseline CNN...")
history1 = model1.fit(
    X_train, y_train_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plot_training_history(history1, 'Model1_Baseline_CNN')

# Evaluate Model 1
metrics1 = evaluate_model(
    model1, X_test, y_test_cat, y_test_indices,
    list(class_indices.keys()), 'Model1_Baseline_CNN'
)

# Visualize predictions
visualize_predictions(model1, X_test, y_test, index_to_class, 'Model1_Baseline_CNN')

# Save model
model1.save(os.path.join(MODEL_DIR, 'model1_baseline_cnn.h5'))


# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model2_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train Model 2 using data augmentation
print("Training Model 2: Advanced CNN with augmentation...")
batch_size = 32
train_generator = data_augmentation.flow(X_train, y_train_cat, batch_size=batch_size)

# Calculate steps per epoch
steps_per_epoch = len(X_train) // batch_size
validation_steps = int(0.2 * len(X_train)) // batch_size

history2 = model2.fit(
    train_generator,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_train[-int(0.2 * len(X_train)):], y_train_cat[-int(0.2 * len(X_train)):]),
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plot_training_history(history2, 'Model2_Advanced_CNN')

# Evaluate Model 2
metrics2 = evaluate_model(
    model2, X_test, y_test_cat, y_test_indices,
    list(class_indices.keys()), 'Model2_Advanced_CNN'
)

# Visualize predictions
visualize_predictions(model2, X_test, y_test, index_to_class, 'Model2_Advanced_CNN')

# Save model
model2.save(os.path.join(MODEL_DIR, 'model2_advanced_cnn.h5'))


# Define callbacks for transfer learning model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model3_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train Model 3 using data augmentation
print("Training Model 3: Transfer Learning with MobileNetV2...")
train_generator = data_augmentation.flow(X_train, y_train_cat, batch_size=batch_size)

history3 = model3.fit(
    train_generator,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_train[-int(0.2 * len(X_train)):], y_train_cat[-int(0.2 * len(X_train)):]),
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plot_training_history(history3, 'Model3_MobileNetV2')

# Fine-tune the model by unfreezing some of the top layers of the base model
print("\nFine-tuning Model 3...")
# Unfreeze the top 20 layers of the MobileNetV2 base
for layer in model3.layers[1].layers[-20:]:
    layer.trainable = True

# Compile with a lower learning rate
model3.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with unfrozen layers
fine_tune_history = model3.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_train[-int(0.2 * len(X_train)):], y_train_cat[-int(0.2 * len(X_train)):]),
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Plot fine-tuning history
plot_training_history(fine_tune_history, 'Model3_MobileNetV2_FineTuned')

# Evaluate Model 3
metrics3 = evaluate_model(
    model3, X_test, y_test_cat, y_test_indices,
    list(class_indices.keys()), 'Model3_MobileNetV2'
)

# Visualize predictions
visualize_predictions(model3, X_test, y_test, index_to_class, 'Model3_MobileNetV2')

# Save model
model3.save(os.path.join(MODEL_DIR, 'model3_mobilenetv2.h5'))


# Collect all metrics
all_metrics = [metrics1, metrics2, metrics3]
df_metrics = pd.DataFrame(all_metrics)

# Display metrics table
print("Model Comparison:")
print(df_metrics.set_index('model_name'))

# Compare metrics visually
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
model_names = [metrics['model_name'] for metrics in all_metrics]

# Bar chart comparison
plt.figure(figsize=(12, 8))
x = np.arange(len(metrics_to_plot))
width = 0.25

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (metrics, color) in enumerate(zip(all_metrics, colors)):
    values = [metrics[metric] for metric in metrics_to_plot]
    plt.bar(x + i*width, values, width, label=metrics['model_name'], color=color, alpha=0.8)

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width, metrics_to_plot)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'))
plt.show()

# Save metrics to CSV
df_metrics.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension


def predict_disease(model, img_path, class_indices):
    """Make prediction for a single image"""
    # Load and preprocess image
    img = load_and_preprocess_image(img_path)

    # For MobileNetV2, we need to rescale from [0,1] to [-1,1]
    if 'mobilenetv2' in model.name.lower():
        img = img * 2.0 - 1.0

    # Make prediction
    pred = model.predict(img)
    pred_class_idx = np.argmax(pred[0])
    pred_prob = pred[0][pred_class_idx]

    # Map index to class name
    index_to_class = {v: k for k, v in class_indices.items()}
    pred_class = index_to_class[pred_class_idx]

    return pred_class, pred_prob, pred[0]


# Sample inference on a few test images
def sample_inference(model_path, test_img_paths, class_indices):
    """Run sample inference and visualize results"""
    # Load model
    model = load_model(model_path)

    plt.figure(figsize=(15, 4 * len(test_img_paths)))
    for i, img_path in enumerate(test_img_paths):
        # Make prediction
        pred_class, pred_prob, all_probs = predict_disease(model, img_path, class_indices)

        # Load image for display
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image and prediction
        plt.subplot(len(test_img_paths), 2, i * 2 + 1)
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class}\nConfidence: {pred_prob:.2%}")
        plt.axis('off')

        # Display probability distribution
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


# Select a few test images for inference
# We'll randomly select 3 images from the test set
test_img_paths = []
for cls in class_indices.keys():
    test_dir = os.path.join(DATA_DIR, 'test', cls)
    if os.path.exists(test_dir):
        img_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if img_files:
            test_img_paths.append(os.path.join(test_dir, random.choice(img_files)))

# Run sample inference with the best model (Model 3)
best_model_path = os.path.join(MODEL_DIR, 'model3_mobilenetv2.h5')
sample_inference(best_model_path, test_img_paths[:3], class_indices)