from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the CSV file and train the model upon starting the server
data = pd.read_csv('malicious_phish.csv')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['url'])
y = data['type']

# Handling class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the entire dataset
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Classification report
report = classification_report(y_test, predictions)
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
    # Uncomment the following line to see the raw prediction value
    # print("Raw Prediction:", rf_model.predict_proba(input_features))
    prediction = rf_model.predict(input_features)[0]
    
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

