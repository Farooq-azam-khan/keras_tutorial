
import tensorflow as tf 
import string 
import re 
from tensorflow.keras import layers 

print(tf.__version__)
print(f'Number of GPU\'s available:', len(tf.config.list_physical_devices('GPU')))

batch_size = 32 
validation_split = 0.2 
seed = 1337

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=validation_split, 
    subset='training', 
    seed=seed,
)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=validation_split, 
    subset='validation', 
    seed=seed
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size = batch_size 
)

print(f'Number of batches in raw_train_ds: {raw_train_ds.cardinality()}')
print(f'Number of batches in raw_val_ds: {raw_val_ds.cardinality()}')
print(f'Number of batches in raw_test_ds: {raw_test_ds.cardinality()}')

# normal text standardizer does not strip HTML tags from text. 
# custom standardization function needs to be created 
def custom_standardization(input_data): 
    lowercase = tf.strings.lower(input_data) 
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ') 
    return tf.strings.regex_replace(
        stripped_html, f'[{re.escape(string.punctuation)}]', ''
    )

from tensorflow.keras.layers import TextVectorization 


# model constants 
max_features = 20_000 
embedding_dim = 128 
sequence_length = 500 

# after standardization we can instantiate out text vectorization layer 
# using this layer to normalize, split, and map strings to integers thus setting out `output_mode` to `int`
# setting an explicit maximum sequence length since CNN layers will not support ragged sequences 
vectorize_layer = TextVectorization(
    standardize=custom_standardization, 
    max_tokens=max_features, 
    output_mode='int', 
    output_sequence_length=sequence_length
)

# text only dataset 
text_ds = raw_train_ds.map(lambda x,y:x)
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label): 
    text = tf.expand_dims(text, -1) 
    return vectorize_layer(text), label 

# Vectorize the data 
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU 
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10) 
test_ds = test_ds.cache().prefetch(buffer_size=10)



inputs = tf.keras.Input(shape=(None,), dtype='int64')

# map vocab indicies into a space of dimensionality 
x = layers.Embedding(max_features, embedding_dim)(inputs) 
x = layers.Dropout(0.5)(x) 

# Conv1D + global max pooling (1D convnet becuase it is text data ) 
x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
x = layers.GlobalAveragePooling1D()(x) 

# vanilla hidden layer 
x = layers.Dense(128, activation='relu')(x) 
x = layers.Dense(0.5)(x) 

# project onto a single unit output layer and squash it with a sigmoid ([0,1] range i.e. probability of it being positive or negative)
predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x) 
model = tf.keras.Model(inputs, predictions, name='text_classification') 
print(model.summary())


# binary crossentropy and adam optimizer 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Started training...')
epochs = 3 
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

print(
    model.evaluate(test_ds)
)


