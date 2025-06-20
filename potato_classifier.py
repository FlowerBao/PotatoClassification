import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Step 1: Setup dataset path and image size
image_size = 64
dataset_path = r"C:\Users\DANIEL AMIR\Downloads\Potato disease classification\Potato disease classification\Potato Image\Potato Image"

# Step 2: Load and preprocess images
def load_images(folder, label):
    images = []
    labels = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                images.append(img)
                labels.append(label)
    return images, labels

d_imgs, d_labels = load_images(os.path.join(dataset_path, "D-potato-Output"), 0)
h_imgs, h_labels = load_images(os.path.join(dataset_path, "H-potato-output"), 1)

X = np.array(d_imgs + h_imgs) / 255.0
y = np.array(d_labels + h_labels)

# Step 3: Train CNN ONCE on full dataset
print("\n🧠 Training CNN on the FULL dataset...")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set epoch based on dataset size
epochs = 30 if len(X) < 500 else 20
cnn_model.fit(X, y, epochs=epochs, batch_size=16, verbose=1)

# Step 4: Define model evaluation for SVM & Logistic Regression
def evaluate_classic_models(test_size_ratio):
    print(f"\n🔍 Evaluating traditional models with test size split: {test_size_ratio}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, stratify=y, random_state=42)
    results = []

    # Flatten images for classic models
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # SVM
    svm_model = SVC()
    svm_model.fit(X_train_flat, y_train)
    svm_preds = svm_model.predict(X_test_flat)
    results.append(("SVM", test_size_ratio, y_test, svm_preds))

    # Logistic Regression
    log_model = LogisticRegression(max_iter=2000)  # Increase max_iter to ensure convergence
    log_model.fit(X_train_flat, y_train)
    log_preds = log_model.predict(X_test_flat)
    results.append(("Logistic Regression", test_size_ratio, y_test, log_preds))

    return results

# Step 5: Run evaluations
splits = [0.3, 0.4, 0.5, 0.6, 0.7]
final_results = []

# CNN prediction also needs to be evaluated on same test sets
for s in splits:
    print(f"\n📊 Split {s*100}%: Evaluating models")
    res = evaluate_classic_models(test_size_ratio=s)

    # Evaluate CNN on same test set
    _, X_test_cnn, _, y_test_cnn = train_test_split(X, y, test_size=s, stratify=y, random_state=42)
    cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    res.append(("CNN", s, y_test_cnn, cnn_preds))

    # Save results
    for algo_name, split, true_y, pred_y in res:
        acc = accuracy_score(true_y, pred_y)
        prec = precision_score(true_y, pred_y)
        rec = recall_score(true_y, pred_y)
        f1 = f1_score(true_y, pred_y)
        err = 1 - acc
        final_results.append({
            "Split": f"{int((1-s)*100)}-{int(s*100)}",
            "Algorithm": algo_name,
            "Error": round(err, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4)
        })

# Step 6: Print results in table format
print("\n✅ Final Results:\n")
print(f"{'Split':<10}{'Algorithm':<20}{'Error':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-score'}")
for res in final_results:
    print(f"{res['Split']:<10}{res['Algorithm']:<20}{res['Error']:<10}{res['Accuracy']:<10}{res['Precision']:<10}{res['Recall']:<10}{res['F1-score']}")

print("\n✅ All evaluations completed successfully.")
