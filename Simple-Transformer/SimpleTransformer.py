import os
import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def load_data(file1, file2):
    with io.open(file1, mode='r', encoding='utf-8') as f1, io.open(file2, mode='r', encoding='utf-8') as f2:
        english_sentences = [line.strip() for line in f1]
        german_sentences = ["<start> " + line.strip() + " <end>" for line in f2]

    return english_sentences, german_sentences

def preprocess_data(english_sentences, german_sentences, max_seq_length=None):
    def tokenize_and_pad(sentences, max_len=None):
        tokenizer = Tokenizer(filters='', lower=False)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)

        if max_len is None:
            max_len = max(len(seq) for seq in sequences)

        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

        return tokenizer, padded_sequences

    english_tokenizer, tokenized_english = tokenize_and_pad(english_sentences)
    max_len = max_seq_length if max_seq_length is not None else max(len(seq) for seq in tokenized_english)
    german_tokenizer, tokenized_german = tokenize_and_pad(german_sentences, max_len)

    return (english_tokenizer, tokenized_english), (german_tokenizer, tokenized_german), max_len



def split_data(tokenized_english, tokenized_german, train_ratio):

    # Randomize the order of sentences while maintaining correspondence
    combined_data = list(zip(tokenized_english, tokenized_german))
    np.random.shuffle(combined_data)
    shuffled_english, shuffled_german = zip(*combined_data)

    # Split the data into training and validation sets
    split_index = int(len(shuffled_english) * train_ratio)
    train_english, val_english = shuffled_english[:split_index], shuffled_english[split_index:]
    train_german, val_german = shuffled_german[:split_index], shuffled_german[split_index:]

    return train_english, train_german, val_english, val_german

def create_optimizer(learning_rate):


    # Create the Adam optimizer with the given learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return optimizer

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, hidden_units):
        super(PositionalEncoding, self).__init__()
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.positional_encoding = self._create_positional_encoding(input_shape[1])

    def call(self, inputs):
        return inputs + self.positional_encoding

    def _create_positional_encoding(self, length):
        positional_encoding = np.zeros((1, length, self.hidden_units), dtype=np.float32)
        for pos in range(length):
            for i in range(0, self.hidden_units, 2):
                angle = pos / np.power(10000, (2 * i) / self.hidden_units)
                positional_encoding[:, pos, i] = np.sin(angle)
                positional_encoding[:, pos, i + 1] = np.cos(angle)
        return tf.constant(positional_encoding)
    
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, attention_heads, hidden_units, dropout_rate,
                 input_vocab_size, target_vocab_size, max_seq_length, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.encoder_inputs = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="encoder_inputs")
        self.decoder_inputs = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="decoder_inputs")
        
        # Encoder
        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, self.hidden_units)
        self.encoder_positional_encoding = PositionalEncoding(self.hidden_units)
        self.encoder_layers = []
        for _ in range(self.num_layers):
            self.encoder_layers.extend([
                layers.MultiHeadAttention(num_heads=self.attention_heads, key_dim=self.hidden_units),
                Dropout(self.dropout_rate),
                LayerNormalization(epsilon=1e-6),
                Dense(self.hidden_units * 4, activation="relu"),
                Dropout(self.dropout_rate),
                Dense(self.hidden_units),
                LayerNormalization(epsilon=1e-6)
            ])

        # Decoder
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, self.hidden_units)
        self.decoder_positional_encoding = PositionalEncoding(self.hidden_units)
        self.decoder_layers = []
        for _ in range(self.num_layers):
            self.decoder_layers.extend([
                layers.MultiHeadAttention(num_heads=self.attention_heads, key_dim=self.hidden_units),
                Dropout(self.dropout_rate),
                LayerNormalization(epsilon=1e-6),
                layers.MultiHeadAttention(num_heads=self.attention_heads, key_dim=self.hidden_units),
                Dropout(self.dropout_rate),
                LayerNormalization(epsilon=1e-6),
                Dense(self.hidden_units * 4, activation="relu"),
                Dropout(self.dropout_rate),
                Dense(self.hidden_units),
                LayerNormalization(epsilon=1e-6)
            ])

        # Output layer
        self.output_layer = Dense(target_vocab_size) # The softmax activation was removed.

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder = self.encoder_embedding(encoder_inputs)
        encoder = self.encoder_positional_encoding(encoder)

        for i in range(0, len(self.encoder_layers), 7):
            attention_output = self.encoder_layers[i](query=encoder, key=encoder, value=encoder)
            attention_output = self.encoder_layers[i+1](attention_output)
            encoder = layers.Add()([encoder, attention_output])
            encoder = self.encoder_layers[i+2](encoder)

            mlp_output = self.encoder_layers[i+3](encoder)
            mlp_output = self.encoder_layers[i+4](mlp_output)
            mlp_output = self.encoder_layers[i+5](mlp_output)
            encoder = layers.Add()([encoder, mlp_output])

        decoder = self.decoder_embedding(decoder_inputs)
        decoder = self.decoder_positional_encoding(decoder)

        for i in range(0, len(self.decoder_layers), 10):
            attention_output = self.decoder_layers[i](query=decoder, key=decoder, value=decoder)
            attention_output = self.decoder_layers[i+1](attention_output)
            decoder = layers.Add()([decoder, attention_output])
            decoder = self.decoder_layers[i+2](decoder)

            attention_output = self.decoder_layers[i+3](query=decoder, key=encoder, value=encoder)
            attention_output = self.decoder_layers[i+4](attention_output)
            decoder = layers.Add()([decoder, attention_output])
            decoder = self.decoder_layers[i+5](decoder)

            mlp_output = self.decoder_layers[i+6](decoder)
            mlp_output = self.decoder_layers[i+7](mlp_output)
            mlp_output = self.decoder_layers[i+8](mlp_output)
            decoder = layers.Add()([decoder, mlp_output])

        outputs = self.output_layer(decoder)
        return outputs
    
def apply_gradient_clipping(optimizer, model, loss_object, clip_norm=1.0):
    original_apply_gradients = optimizer.apply_gradients

    def apply_gradients_with_clipping(grads_and_vars, *args, **kwargs):
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads_and_vars]
        return original_apply_gradients(clipped_grads_and_vars, *args, **kwargs)

    optimizer.apply_gradients = apply_gradients_with_clipping
    return optimizer

def configure_learning_rate_scheduler(optimizer, warmup_steps):
    # Set the initial learning rate
    initial_learning_rate = 0.001

    # Set the maximum learning rate
    max_learning_rate = 0.01

    # Define the linear learning rate scheduler with warmup
    def scheduler(epoch, lr):
        if epoch < warmup_steps:
            return initial_learning_rate + (epoch * (max_learning_rate - initial_learning_rate) / warmup_steps)
        else:
            return max_learning_rate - (epoch - warmup_steps) * (max_learning_rate - initial_learning_rate) / (total_epochs - warmup_steps)

    # Create the learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)

    return learning_rate_scheduler


def print_variable(variable_name, variable_value):
    print(f"{variable_name}: {variable_value}, Type: {type(variable_value)}")

def train_model(model, train_english, train_german, val_english, val_german, epochs, optimizer, scheduler):
    print_variable("model", model)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        #print("\ntar before: ", tar.shape)
        tar_inp = tar[:, :-1]
        #print("\ntar input: ", tar_inp.shape)
        tar_real = tar[:, 1:]
        #print("\n input shape: ", inp.shape)

        with tf.GradientTape() as tape:
            predictions = model((inp, tar_inp), training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    checkpoint_dir = './model_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    batch_size = 64  # You can change this value according to your available resources
    print("\nBatch size is: ", batch_size)
    
    print("\Defining datasets (train, val)\n")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_english, train_german)).shuffle(len(train_english)).batch(batch_size, drop_remainder=True)
    print("Dataset shuffled\n")
    val_dataset = tf.data.Dataset.from_tensor_slices((val_english, val_german)).batch(batch_size, drop_remainder=True)


    print("\nThe training begins\n")
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    final_model_name = "SimpleTransformerWeights"
    model.save_weights(final_model_name)

    return model


print("The main function final\n\n\n")
# Load, preprocess, and split data
english_sentences, german_sentences = load_data("/kaggle/input/training-datasets/deu_english.txt", "/kaggle/input/training-datasets/deu_german.txt")
max_seq_length = None
(english_tokenizer, tokenized_english), (german_tokenizer, tokenized_german), max_len = preprocess_data(english_sentences, german_sentences, max_seq_length)
print("\n Max sequence length: ", max_len, "\n")
train_ratio = 1
train_english, train_german, val_english, val_german = split_data(tokenized_english, tokenized_german, train_ratio)

# Convert the train_english, train_german, val_english, and val_german data to tensors
train_english = tf.convert_to_tensor(np.array(train_english))
train_german = tf.convert_to_tensor(np.array(train_german))
val_english = tf.convert_to_tensor(np.array(val_english))
val_german = tf.convert_to_tensor(np.array(val_german))


# Initialize the transformer model
num_layers = 2
attention_heads = 8
hidden_units = 64
dropout_rate = 0.1
input_vocab_size = len(english_tokenizer.word_index) + 1
target_vocab_size = len(german_tokenizer.word_index) + 1

model = TransformerModel(num_layers, attention_heads, hidden_units, dropout_rate, input_vocab_size, target_vocab_size, max_seq_length)

# Create and configure the optimizer with gradient clipping and learning rate scheduler
learning_rate = 0.001
optimizer = create_optimizer(learning_rate)
optimizer_with_clipping = apply_gradient_clipping(optimizer, model, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none"))
warmup_steps = 3
scheduler = configure_learning_rate_scheduler(optimizer_with_clipping, warmup_steps)

# Train the model
epochs = 10
trained_model = train_model(model, train_english, train_german, val_english, val_german, epochs, optimizer, scheduler)

# Save max_len, english_tokenizer, and german_tokenizer
with open('max_len.pkl', 'wb') as f:
    pickle.dump(max_len, f)

with open('english_tokenizer.pkl', 'wb') as f:
    pickle.dump(english_tokenizer, f)

with open('german_tokenizer.pkl', 'wb') as f:
    pickle.dump(german_tokenizer, f)
    
def translate_sentence(model, input_sentence, german_tokenizer, max_len):
    # Pad the input_sentence to max_len
    padded_sentence = pad_sequences(input_sentence, maxlen=max_len, padding='post', truncating='post')

    # Encode the input sentence
    input_sequence = tf.convert_to_tensor(np.array(padded_sentence))  # Add batch dimension

    # Initialize the decoder input with the start token and pad it
    decoder_input_tokens = [german_tokenizer.word_index['<start>']]
    decoder_input_padded = pad_sequences([decoder_input_tokens], maxlen=max_len - 1, padding='post', truncating='post')
    decoder_input = tf.convert_to_tensor(decoder_input_padded)

    # Print the shape of input_sequence and decoder_input
    print("Input sequence shape:", input_sequence.shape)
    print("Decoder input shape:", decoder_input.shape)

    # Initialize the output German sentence
    output_sentence = []

    for _ in range(max_len):
        # Make a prediction
        predictions = model((input_sequence, decoder_input), training=False)

        # Get the index of the most probable token
        predicted_token_index = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]

        # Check if the predicted token is the end token
        if german_tokenizer.index_word[predicted_token_index] == '<end>':
            break

        # Add the predicted token to the output sentence
        output_sentence.append(german_tokenizer.index_word[predicted_token_index])

        # Update the decoder input with the predicted token
        decoder_input_tokens.append(predicted_token_index)
        decoder_input_padded = pad_sequences([decoder_input_tokens], maxlen=max_len - 1, padding='post', truncating='post')
        decoder_input = tf.convert_to_tensor(decoder_input_padded)

    # Convert the output sentence to German text
    output_text = ' '.join(output_sentence)

    print("Translated sentence:", output_text)
    
# Tokenize and pad a sample English sentence
english_sentence = "I love it."
tokenized_sentence = english_tokenizer.texts_to_sequences([english_sentence])

# Translate the sentence
translate_sentence(trained_model, tokenized_sentence, german_tokenizer, max_len)
