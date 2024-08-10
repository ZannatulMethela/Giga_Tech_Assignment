# Giga_Tech_Assignment
#NLP Pipeline with POS and NER Tagging

## Overview
This project involves training a BiLSTM model for Part-of-Speech (POS) tagging and Named Entity Recognition (NER) using a dataset in Bangla. The notebook covers data preprocessing, model training, evaluation, and making predictions on new data.

## Table of Contents
1. Requirements
2. Setup
3. Running the Pipeline
4. Model Deployment
5. Making Predictions on New Data

## Requirements
To run this project, you'll need the following dependencies:

- TensorFlow
- scikit-learn
- pandas
- numpy

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Setup
1. **Clone the Repository:**
   ```bash
   git clone https://ZannatulMethela/Giga_Tech_Assignment.git
   cd Giga_Tech_Assignment
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.7+ installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset:**
   Place your dataset (`data.tsv`) in the root directory or modify the notebook to point to your dataset location.

## Running the Pipeline
1. **Data Preprocessing:**
   The first section of the notebook handles data preprocessing, including tokenization, label encoding, and sequence padding.

2. **Model Training:**
   The next section defines and trains a BiLSTM model for POS tagging and NER tasks.

3. **Model Evaluation:**
   After training, the model is evaluated on a test set to check its accuracy on POS and NER tagging.

## Model Deployment
1. **Save the Model:**
   You can save the trained model for deployment:
   ```python
   model.save('pos_ner_model.h5')
   ```

2. **Load the Model:**
   To use the model for predictions:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('pos_ner_model.h5')
   ```

## Making Predictions on New Data
1. **Preprocess New Data:**
   Use the tokenizer and sequence padding from the preprocessing section to prepare new sentences for prediction.

2. **Make Predictions:**
   ```python
   new_sentence = "your sentence here"
   # Tokenize and pad the new sentence
   new_sequence = tokenizer.texts_to_sequences([new_sentence])
   new_sequence = pad_sequences(new_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

   # Predict POS and NER tags
   predictions = model.predict(new_sequence)
   pos_tags, ner_tags = predictions[0], predictions[1]
   ```

3. **Interpret Results:**
   Convert the predicted labels back to their original form using the encoders:
   ```python
   predicted_pos = pos_encoder.inverse_transform(np.argmax(pos_tags, axis=-1)[0])
   predicted_ner = ner_encoder.inverse_transform(np.argmax(ner_tags, axis=-1)[0])
   ```

