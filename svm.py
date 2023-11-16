import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the CSV file
data = pd.read_csv('malicious_phish.csv')

# Feature Extraction using TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['url'])
y = data['type']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Splitting the training set into smaller chunks for partial fit
batch_size = 1000
sgd_model = SGDClassifier(loss='hinge', random_state=42)

# Get the number of batches needed
num_samples = X_train.shape[0]  # Get the number of samples (rows) in the matrix
num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches needed

# Iteratively train the model on smaller batches
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]
    sgd_model.partial_fit(X_batch, y_batch, classes=np.unique(y))

# Evaluate the model on the testing set
predictions = sgd_model.predict(X_test)

# Classification report
report = classification_report(y_test, predictions)
print(report)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)



