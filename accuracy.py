import pandas as pd

# Load the CSV file
data = pd.read_csv('malicious_phish.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'URL' column contains URLs and 'Type' column contains categories
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['url'])
y = data['type']  # Target variable

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Create RandomForest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)


