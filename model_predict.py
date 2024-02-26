# -*- coding: utf-8 -*-

import os, json
import numpy as np
from albert_zh.extract_feature import BertVector
from keras.models import load_model
from att import Attention

# Load the best trained model
model_dir = './models'
files = os.listdir(model_dir)
models_path = [os.path.join(model_dir, _) for _ in files]
for i in models_path:
    print(i)
best_model_path = sorted(models_path, key=lambda x: float(x.split('-')[-1].replace('.h5', '')), reverse=True)[0]
print("best_model_path："+best_model_path)
model = load_model(best_model_path, custom_objects={"Attention": Attention})

# Sample Statements and Preprocessing
#As the study data are in Chinese, the examples here are also in Chinese.
text1 = '湖北#丘陵#湖北大多数地区都属于丘陵地带。'
#Translate to:
#text1 = 'Hubei #Hills #Most of Hubei is hilly.'

per1, per2, doc = text1.split('#')
text = '$'.join([per1, per2, doc.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
print(text)


# Sentence feature extraction using BERT
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)#128
vec = bert_model.encode([text])["encodes"][0]

x_train = np.array([vec])

# Model predictions and output predictions
predicted = model.predict(x_train)
y = np.argmax(predicted[0])

with open('data/rel_dict.json', 'r', encoding='utf-8') as f:
    rel_dict = json.load(f)

id_rel_dict = {v:k for k,v in rel_dict.items()}
print('original text: %s' % text1)
print('prediction: %s' % id_rel_dict[y])

