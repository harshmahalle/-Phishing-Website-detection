#gradient boosting
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the CSV file and train the model upon starting the server
data = pd.read_csv('malicious_phish.csv')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['url'])
y = data['type']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical labels to numerical values
y_encoded = label_encoder.fit_transform(y)

# Handling class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.toarray(), y_encoded)

# Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Convert test labels to encoded values using the same label encoder
y_test_encoded = label_encoder.transform(y_test)

# Make predictions on the test set
predictions = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, predictions)
print("Accuracy:", accuracy)

# Classification report
report = classification_report(y_test_encoded, predictions)
print(report)

# Check class distribution
class_distribution = data['type'].value_counts()
print(class_distribution)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    input_features = vectorizer.transform([url])
    prediction = xgb_model.predict(input_features)[0]
    
    if prediction == 0:
        result = "Safe"
    elif prediction == 1:
        result = "Malware"
    elif prediction == 2:
        result = "Phishing"
    else:
        result = "Defacement"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
