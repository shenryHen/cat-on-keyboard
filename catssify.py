import collections
import pathlib
import re
import string

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['data/catInput.csv', 'data/humanInput.csv']

# for name in FILE_NAMES:
#   text_dir = utils.get_file(name, origin=name)

# parent_dir = pathlib.Path(text_dir).parent
# list(parent_dir.iterdir())

def labeler(example, index):
  return example, tf.cast(index, tf.int64)
labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(str(file_name))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 4
VALIDATION_SIZE = 4

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

for text, label in all_labeled_data.take(10):
  print("Sentence: ", text.numpy())
  print("Label:", label.numpy())

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

#tokenized_ds = configure_dataset(tokenized_ds)

vocab_dict = collections.defaultdict(lambda: 0)
for toks in all_labeled_data.as_numpy_iterator():
  for tok in toks:
    vocab_dict[tok] += 1

vocab_dict.pop(0)
vocab_dict.pop(1)
#print(vocab_dict)
VOCAB_SIZE = 10000
vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for token, count in vocab]
vocab = vocab[:VOCAB_SIZE]
vocab_size = len(vocab)
print("Vocab size: ", vocab_size)
#print("First five vocab entries:", vocab[:5])

keys = vocab
values = range(2, len(vocab) + 2)  # reserve 0 for padding, 1 for OOV

init = tf.lookup.KeyValueTensorInitializer(
    keys, values, key_dtype=tf.string, value_dtype=tf.int64)

num_oov_buckets = 1
vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)

def preprocess_text(text, label):
  standardized = text
  tokenized = text
  vectorized = vocab_table.lookup(tokenized)
  return vectorized, label

example_text, example_label = next(iter(all_labeled_data))
print("Sentence: ", example_text.numpy())
vectorized_text, example_label = preprocess_text(example_text, example_label)
print("Vectorized sentence: ", vectorized_text.numpy())

all_encoded_data = all_labeled_data.map(preprocess_text)

train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
validation_data = all_encoded_data.take(VALIDATION_SIZE)

train_data = train_data.padded_batch(BATCH_SIZE)
validation_data = validation_data.padded_batch(BATCH_SIZE)

sample_text, sample_labels = next(iter(validation_data))
print("Text batch shape: ", sample_text.shape)
print("Label batch shape: ", sample_labels.shape)
print("First text example: ", sample_text[0])
print("First label example: ", sample_labels[0])

vocab_size += 2

def create_model(vocab_size, num_labels):
  model = tf.keras.Sequential([
      layers.Embedding(vocab_size, 64, mask_zero=True),
      layers.Conv1D(16, 1, padding="valid", activation="relu", strides=2),
      layers.GlobalMaxPooling1D(),
      layers.Dense(num_labels)
  ])
  return model

model = create_model(vocab_size=vocab_size, num_labels=3)
model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
history = model.fit(train_data, validation_data=validation_data, epochs=5)

loss, accuracy = model.evaluate(validation_data)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
    split=' ', char_level=False, oov_token=None, document_count=0
)

MAX_SEQUENCE_LENGTH = 250

MAX_SEQUENCE_LENGTH = 250

preprocess_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

preprocess_layer.set_vocabulary(vocab)

export_model = tf.keras.Sequential(
    [preprocess_layer, model,
     layers.Activation('sigmoid')])

export_model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])
# Create a test dataset of raw strings
AUTOTUNE = tf.data.experimental.AUTOTUNE 
test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
test_ds = configure_dataset(test_ds)
loss, accuracy = export_model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))

inputs = [
    "Don\'t let the clipperl blowing a three one lead distract you from the fact that the lakers won the finals",  # Label: 1
    "'a a s s e e r r o s o",  # Label: 2
    "z z z z z z z z z z z z z z z z z z z z z  z z z z z z z  right alt",  # Label: 0
]
predicted_scores = export_model.predict(inputs)
predicted_labels = tf.argmax(predicted_scores, axis=1)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy())

model.save('terribleCatModel.h5')

from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

@app.route('/cat', methods=['GET', 'POST'])
def cat():
  if request.method == 'GET':
    return render_template(' base.html')
  if request.method == 'POST':
    curr_input = request.form['input']
    predicted_scores = export_model.predict([curr_input])
    predicted_labels = tf.argmax(predicted_scores, axis=1)
    if int(predicted_labels) == 1:
      return 'cat'
    else:
      return 'human'
    #print(request.get_json())

@app.route('/')
def index():
  return render_template('base.html')