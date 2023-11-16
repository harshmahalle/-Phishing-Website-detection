from sklearn.datasets import fetch_openml

# Load the phishing websites dataset from Scikit-learn
phishing_data = fetch_openml(name='phishing')

# Display information about the dataset
print("Dataset Information:")
print(phishing_data.DESCR)  # Description of the dataset
print("\nDataset Features:")
print(phishing_data.feature_names[:10])  # Display the first 10 feature names
print("\nTarget Classes:")
print(phishing_data.target_names)  # Display the target variable names


