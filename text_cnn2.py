import pandas as pd
import matplolib as plt
from sklearn.model_selection import train_test_split
import transformers
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, \
    DistilBertConfig, DistilBertTokenizerFast
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

dataset = pd.read_csv('./data/train_csv/policies_train.csv')
texts = list(dataset["text"])
label_names = dataset.drop(["source", 'text'], axis=1).columns
labels = dataset[label_names].values

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

sample_idx = 23
print(f'Sample: "{train_texts[sample_idx]}"')
print(f"Labels: {pd.Series(train_labels[sample_idx], label_names).to_dict()}")


MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 200  # We truncate anything after the 200-th word to speed up training

# The configuration is not needed if you don't have to customize the
# network architecture. Here we will need it to replacee the output of the model
# with a multi-label prediction layer (i.e. sigmoid activations + binary cross-entropy
# instead of softmax + categorical cross-entropy of multi-class classification)
config = DistilBertConfig.from_pretrained(MODEL_NAME)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                            max_length=MAX_LENGTH, return_tensors="tf")
test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                           max_length=MAX_LENGTH, return_tensors="tf")

# Create TensorFlow datasets to feed the model for training and evaluation
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

# Tokenizer output example
sample_text = "I have changed the headers to small letters, since I was basically..."
tokenizer.decode(tokenizer(sample_text)["input_ids"])


transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, output_hidden_states=False
)

bert = transformer_model.layers[0]

# The input is a dictionary of word identifiers
input_ids = Input(shape=(MAX_LENGTH,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}

# Here we select the representation of the first token ([CLS]) for classification
# (a.k.a. "pooled representation")
bert_model = bert(inputs)[0][:, 0, :]

# Add a dropout layer and the output layer
dropout = Dropout(config.dropout, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
output = Dense(
    units=train_labels.shape[1],
    kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
    activation="sigmoid",  # Choose a sigmoid for multi-label classification
    name='output'
)(pooled_output)

model = Model(inputs=inputs, outputs=output, name='BERT_MultiLabel')
model.summary()


def multi_label_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """For multi-label classification, one has to define a custom
    acccuracy function because neither tf.keras.metrics.Accuracy nor
    tf.keras.metrics.CategoricalAccuracy evaluate the number of
    exact matches.

    :Example:
    >>> from tensorflow.keras import metrics
    >>> y_true = tf.convert_to_tensor([[1., 1.]])
    >>> y_pred = tf.convert_to_tensor([[1., 0.]])
    >>> metrics.Accuracy()(y_true, y_pred).numpy()
    0.5
    >>> metrics.CategoricalAccuracy()(y_true, y_pred).numpy()
    1.0
    >>> multi_label_accuracy(y_true, y_pred).numpy()
    0.0
    """
    y_pred = tf.math.round(y_pred)
    exact_matches = tf.math.reduce_all(y_pred == y_true, axis=1)
    exact_matches = tf.cast(exact_matches, tf.float32)
    return tf.math.reduce_mean(exact_matches)


loss = BinaryCrossentropy()
optimizer = Adam(5e-5)
metrics = [
    multi_label_accuracy,
    "binary_accuracy",
    AUC(name="average_precision", curve="PR", multi_label=True)
]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
training_history = model.fit(
    train_dataset.shuffle(1000).batch(16), epochs=2, batch_size=16,
    validation_data=test_dataset.batch(16)
)


# evaluate the model
benchmarks = model.evaluate(test_dataset.batch(16), return_dict=True, batch_size=16)

print(benchmarks)

loss, accuracy = model.evaluate(test_dataset.batch(16))

print("Loss: ", loss)
print("Accuracy: ", accuracy)
