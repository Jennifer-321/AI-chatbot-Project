import os
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import layers

# Get the full path to the intents file
intents_path = os.path.join(os.path.dirname(__file__), 'data', 'intents.json')

# Load the intents data from the file
with open(intents_path, 'r') as file:
    intents = json.load(file)

# Load intents from intents.json file
#with open('intents.json', 'r') as file:
 #   intents = json.load(file)

# Preprocess text using NLTK
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training_data = []
output_data = []

output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training_data.append(bag)
    output_data.append(output_row)

training_data = np.array(training_data)
output_data = np.array(output_data)

# Build neural network model using TensorFlow
model = tf.keras.Sequential([
    layers.Dense(64, input_shape=[len(words)]),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, output_data, epochs=100)

# Save trained model using pickle
with open('chatbot_model.pickle', 'wb') as file:
    pickle.dump(model, file)


    def clean_text(text, lemmatizer, ignore_letters):
        """
        Cleans and preprocesses the user input text.

        :param text: str, the user input text to preprocess.
        :param lemmatizer: an instance of the WordNetLemmatizer class from the NLTK library.
        :param ignore_letters: list of str, containing letters to ignore in the text.
        :return: list of str, containing the preprocessed words from the text.
        """
        cleaned_text = ''
        for letter in text.lower():
            if letter not in ignore_letters:
                cleaned_text += letter

        tokenized_text = nltk.word_tokenize(cleaned_text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_text]

        return lemmatized_words


    # Use trained model to make predictions
    def predict_class(input_text):
        input_text = clean_text(input_text, lemmatizer, ignore_letters)
        bag = [0] * len(words)
        for word in input_text:
            for i, w in enumerate(words):
                if w == word:
                    bag[i] = 1

        prediction = model.predict(np.array([bag]))[0]
        max_value_index = np.argmax(prediction)
        tag = classes[max_value_index]

        if prediction[max_value_index] < 0.5:
            tag = 'unknown'

        return tag


# Generate response based on predicted class
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Run the chatbot
print("Chat with the bot (type 'quit' to exit):")
while True:
    # Get user input
    user_input = input("You: ")

    # Check if user wants to quit
    if user_input.lower() == 'quit':
        break

    # Get prediction from the model
    predicted_tag = predict_class(user_input)

    # Get response from intents
    response = get_response(intents_list=[{'intent': predicted_tag, 'probability': 1.0}], intents_json=intents)

    # Print the response
    print("Bot:", response)