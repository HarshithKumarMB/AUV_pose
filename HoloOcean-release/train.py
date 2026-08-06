import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import joblib

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

df1 = pd.read_csv("/home/harshith/Downloads/estimated_3d_obstacle_coordinates_1m.csv")

df2 = pd.read_csv("/home/harshith/Downloads/seabed_coordinates_1m_labeled.csv")


df = pd.concat([df1, df2], ignore_index=True)

#print("Dataset 1 shape:", df1.shape)
#print("Dataset 2 shape:", df2.shape)
print("Combined shape:", df.shape)



# =====================================================
# FEATURES AND LABELS
# =====================================================

X = df[["x", "y", "z"]]

y = df["obstacle_type"]

# =====================================================
# ENCODE LABELS
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
print(df["obstacle_type"].value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================================
# MODEL
# =====================================================

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "obstacle_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel saved successfully.")