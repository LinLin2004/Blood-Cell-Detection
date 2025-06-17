import os, joblib
from utils.feature_extraction import extract_features_from_dataset
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def train_model(data_dir, save_path, feature_type='hog', model_type='svm'):
    print("[INFO] Extracting training features...")
    X, y = extract_features_from_dataset(data_dir, split='train', feature_type=feature_type)
    print(f"[INFO] Training {model_type.upper()} classifier with {len(X)} samples...")

    if model_type == 'svm':
        clf = svm.SVC(kernel='linear', probability=True)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    clf.fit(X, y)
    joblib.dump(clf, save_path)
    print(f"[INFO] Model saved to {save_path}")

