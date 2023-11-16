# LSTM nueral network
from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

app = Flask(__name__)

# Load the CSV file and train the model upon starting the server
data = pd.read_csv('malicious_phish.csv')

# Tokenize the URLs
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['url'])
sequences = tokenizer.texts_to_sequences(data['url'])
max_sequence_length = 100  # Adjust based on your data
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert target labels to categorical if needed
y_categorical = to_categorical(data['type'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))  # 4 classes: benign, defacement, malware, phishing

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    input_features = tokenizer.texts_to_sequences([url])
    input_features = pad_sequences(input_features, maxlen=max_sequence_length)
    prediction = model.predict(input_features)
    predicted_class = ['Safe', 'Malware', 'Phishing', 'Defacement'][prediction.argmax(axis=-1)[0]]
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
