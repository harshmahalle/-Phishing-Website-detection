from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the CSV file
data = pd.read_csv('malicious_phish.csv')

# Training Word2Vec embeddings on the URLs
url_tokens = [url.split('/') for url in data['url']]
word2vec_model = Word2Vec(url_tokens, vector_size=100, window=5, min_count=1, sg=1)

# Function to average Word2Vec embeddings for URLs
def get_average_word2vec(tokens, vector, generate_missing=False, k=100):
    if len(tokens) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

# Transforming URLs to Word2Vec average embeddings
url_embeddings = [get_average_word2vec(tokens, word2vec_model.wv) for tokens in url_tokens]
X_word2vec = np.array(url_embeddings)

# Assigning X_word2vec and y from the Word2Vec embeddings and labels respectively
X = X_word2vec
y = data['type']

# Handling class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
batch_size = 1000
sgd_model = SGDClassifier(loss='hinge', random_state=42)
num_samples = X_train.shape[0]
num_batches = (num_samples + batch_size - 1) // batch_size

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]
    sgd_model.partial_fit(X_batch, y_batch, classes=np.unique(y))

# Make predictions on the test set
predictions = sgd_model.predict(X_test)

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
    url_tokens = url.split('/')
    input_features = get_average_word2vec(url_tokens, word2vec_model.wv)
    input_features = input_features.reshape(1, -1)  # Reshape for compatibility with the model
    
    # Get the raw prediction values
    raw_predictions = sgd_model.decision_function(input_features)
    print("Raw Predictions:", raw_predictions)
    
    # Convert raw predictions to probabilities using softmax
    probabilities = np.exp(raw_predictions) / np.sum(np.exp(raw_predictions), axis=1, keepdims=True)
    print("Probabilities:", probabilities)
    
    # Determine the predicted class
    predicted_class = np.argmax(probabilities)
    class_labels = ['Safe', 'Malware', 'Phishing', 'Defacement']
    result = class_labels[predicted_class]
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

