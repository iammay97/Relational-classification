# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from att import Attention
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from operator import itemgetter

#input data
from load_data import get_train_test_pd

#import Albert-pre-trained language models
from albert_zh.extract_feature import BertVector

#If you switch to the bert model, use the following commands
#from bert.extract_feature import BertVector

# Reading files and converting them
train_df, test_df = get_train_test_pd()

bert_model = BertVector(pooling_strategy="NONE", max_seq_len=128)## This parameter msl and wordlen should be changed together
print('begin encoding')

f = lambda text: bert_model.encode([text])["encodes"][0]

train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)

print('end encoding')

# Training and test sets

x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])

print('x_train: ', x_train.shape)

# Convert type y values to ont-hot vectors

num_classes = 9
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model Structure: ALBERT + Bidirectional GRU + Attention + FC
# Input(shape=(200, 312, ))
wordlen = 128
#312 for Albert, 768 for Bert.
input_layer = Input(shape=(wordlen,312))
#embedding_layer = Embedding(input_dim=32,output_dim=50)(input_layer)
gru = Bidirectional(LSTM(wordlen,dropout=0.1,return_sequences=True))(input_layer)
# Use Bidirectional with RNN
attention = Attention(32)(gru)
output = Dense(num_classes, activation='softmax')(attention)

#Removing the attention mechanism, the dimensions need to be levelled, and the following line of code is the leveller
#gru=Flatten()(gru)
#output = Dense(num_classes, activation='softmax')(gru)
model = Model(inputs=input_layer,outputs=output)
# Model visualisation
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy']
              )

# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

# If the .h5 file exists in the original mods folder, delete it.
model_dir = './models'
if os.listdir(model_dir):
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

# Save the latest val_acc best model file
filepath="./models/per-rel-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

# Model training, batch_size and epoch parameters can be adjusted.

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=30, callbacks=[early_stopping, checkpoint])
# model.save('people_relation.h5')

print('effect on the test set:', model.evaluate(x_test, y_test))

# Read relationship correspondence table
with open('./data/rel_dict.json', 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())

sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]
print(values)

# Output classification report for each category
y_pred = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=values,digits=4))


# Plotting loss and acc images
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("loss_acc.png")

