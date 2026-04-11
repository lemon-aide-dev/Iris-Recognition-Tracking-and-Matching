import os
import numpy as np
from core_detection import preprocess_the_img, detect_the_iris, extract_the_iris_features

def load_dataset(dataset_path):
    dataset_features = []
    labels = []

    for file in os.listdir(dataset_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.npy')):
            path = os.path.join(dataset_path, file)

            gray = preprocess_the_img(path)
            iris, _ = detect_the_iris(gray)

            if iris is not None:
                features = extract_the_iris_features(iris)
                dataset_features.append(features)
                labels.append(file.split("_")[0])  # cleaner label

    print(f"Loaded {len(dataset_features)} iris samples")
    return dataset_features, labels


def find_best_match(live_features, dataset_features, labels):
    best_score = -1
    best_label = "Unknown"

    for i, features in enumerate(dataset_features):
        score = np.dot(live_features, features)

        if score > best_score:
            best_score = score
            best_label = labels[i]

    if best_score < 0.9:
        best_label = "Unknown"

    return best_label, best_score
