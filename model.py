import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv(r"E:/randsomeware_detection/dataset/Obfuscated-MalMem2022.csv")
df.dropna(inplace=True)

# Create binary label
df['Ransomware_Label'] = df['Category'].apply(lambda x: 1 if 'Ransomware' in x else 0)

# Drop unused columns
df.drop(columns=['Category', 'Class'], inplace=True)

# Features and label
X = df.drop('Ransomware_Label', axis=1)
y = df['Ransomware_Label']

# Split for importance calculation (initial model)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)

# Train initial model to get feature importances
model_all = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_all.fit(X_train_all_scaled, y_train_all)

# Get top 15 features
importances = model_all.feature_importances_
top_features = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15).index.tolist()

# Save top features
joblib.dump(top_features, 'top_features.pkl')

# Now train on only top 15 features
X_top = df[top_features]
y = df['Ransomware_Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train final model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.savefig('confusion_matrix.png')


# Save model and scaler
joblib.dump(model, 'ransomware_model_15.pkl')
joblib.dump(scaler, 'scaler_15.pkl')

# Save top features to a CSV file
top_features_df = pd.DataFrame(top_features, columns=['Top_Features'])
top_features_df.to_csv('top_features.csv', index=False)

print("\nTop 15 features saved to 'top_features.csv'!")

print("\nModel trained and saved using top 15 features!")
