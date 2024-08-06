import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dataset
df = pd.read_csv('C:\\Users\\SURESH PATEL\\Downloads\\iris classification\\data\\iris_flower.csv')

# Check for missing values
if df.isnull().sum().sum() == 0:
    print("No missing values detected.")

# Feature and target separation
X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection and training
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model and scaler
with open('C:\\Users\\SURESH PATEL\\Downloads\\iris classification\\models\\iris_model.pkl', 'wb') as f:
    pickle.dump((scaler, model), f)
