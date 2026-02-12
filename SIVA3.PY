import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Dogs vs Cats Dataset simulation (HOG-like flattened features for SVM, standard approach) [web:25][web:28][web:29]
# 400 samples: 200 cats, 200 dogs; each "image" as 100 features (flattened resized 10x10 grayscale)
np.random.seed(42)
n_samples_per_class = 200
n_features = 100  # Simulated flattened image features (e.g., from 10x10 grayscale)

# Generate synthetic but realistic image features (cats/dogs patterns)
cat_features = np.random.normal(0.4, 0.2, (n_samples_per_class, n_features))  # Cats: mid-low tones
dog_features = np.random.normal(0.6, 0.2, (n_samples_per_class, n_features))  # Dogs: mid-high tones
X = np.vstack([cat_features, dog_features])
y = np.array(['cat'] * n_samples_per_class + ['dog'] * n_samples_per_class)

df = pd.DataFrame(X)
df['label'] = y

print("Dataset preview:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df['label'].value_counts())

# Features and target
X_features = df.drop('label', axis=1)
y_labels = df['label'].map({'cat': 0, 'dog': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, random_state=42, stratify=y_labels)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction with PCA (visualization, SVM handles high-dim) [web:29]
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nExplained variance by first 2 PCs: {pca.explained_variance_ratio_.sum():.3f}")

# Train SVM with RBF kernel (effective for image classification) [web:28][web:32]
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

# Evaluation
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print("\nSVM Model Performance:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Cat', 'Dog']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualization: PCA scatter plot with SVM decision boundary approx [web:29]
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Train Data (Cats vs Dogs)')
plt.colorbar(scatter, ticks=[0, 1], label='Class (0=Cat, 1=Dog)')

# Simple decision boundary visualization on PCA (for illustration)
from sklearn.inspection import DecisionBoundariesDisplay
DecisionBoundariesDisplay.from_estimator(
    svm_model, X_train_pca, cmap='RdYlBu_r', response_method='predict', alpha=0.3
)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k', alpha=0.7)
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.title('SVM Decision Boundary (PCA space)')

plt.tight_layout()
plt.show()

# Example prediction for a new "image"
new_image = np.random.normal(0.5, 0.2, n_features)  # Neutral sample
new_scaled = scaler.transform([new_image])
new_pca = pca.transform(new_scaled)
pred = svm_model.predict(new_scaled)[0]
prob = svm_model.decision_function(new_scaled)[0]  # Distance to hyperplane
print(f"\nExample prediction: {'Dog' if pred == 1 else 'Cat'} (decision: {prob:.2f})")

print("\nModel ready for Kaggle Dogs vs Cats! For real images, replace feature extraction with HOG/OpenCV resize/flatten [web:29][web:32].")
