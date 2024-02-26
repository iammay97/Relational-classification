# **Domain Ontology-Driven ALBERT Model for Relationship Extraction from Landslide Disaster Reports**

---
## **About The Project**
### **_ABSTRACT_**

Knowledge mapping provides knowledge support for geological hazards that consider factors such as complexity, suddenness, and spatial-temporal nature, and relationship extraction is a key part of knowledge mapping construction. However, unstructured data accumulated over the years cannot intuitively display complex geoscientific information, inter-information relationships, and deep connections across texts. In order to solve this problem, this study proposes a framework for the relation extraction of landslide geological disasters based on the domain ontology. Firstly, an ontology of landslide hazard chains is constructed from the collected corpus to define the conceptual architecture of landslide hazards. Then, the deep learning model is applied for the first time to the relationship extraction task, where the ALBERT (A Lite Bert) pre-trained language model is embedded to obtain character vectoring. Its textual features are fed into the BiGRU-Attention relationship extraction model for training. The resulting probabilistic weights are assigned to be summed up with the product of the states of the individual hidden layers to determine the result of the relationship classification. The comprehensive experimental results show that the ALBERT-BiGRU-Attention relationship extraction model performs best, with a Precision rate of 86.40%, a recall rate of 87.88%, and an F1 score of 88.46%. Therefore, the methodology adopted in this study can provide technical support for the construction of a knowledge map of landslide geohazards, and the visualization of the results can demonstrate the process of spatial and temporal changes of landslide hazards as well as the impact on human behavioral activities.

### _**Keywords:**_

Knowledge graph，Landslide geological disaster，Spatial and temporal characteristics，Relationship extraction，Word embedding，Deep learning model

### _Code availability section_

Name of the code:Relational-classification

Contact: maying2019@cug.edu.cn

Program language: python

Software required:  python3.6, Tensorflow1.15, keras2.3.1.

The source codes are available for downloading at the link: https://github.com/... 

### _Data availability_

Data will be made available on request.

This is a Tensorflow implementation of the paper <Domain Ontology-Driven ALBERT Model for Relationship Extraction from Landslide Disaster Reports>

We have provided the trained ALBERT-BIGRU-ATTENTION model (in checkpoint) in this paper (section 4.2), and you can test it with the model_train.py

In addition, we have shared some of the used case datasets in the paper, and you can also train your model using model_train.py by setting up a local data path.

If you have any question, please contact the corresponding author.

Thanks for your attention!

----

## **The following documentation describes the model_train.py and model_predict.py programs**


### **1. Statement:**

This study is based on the Chinese dataset, the data content is in the `data_example.xls` file, and only 100 test data are provided due to data confidentiality issues.
 
_Raw data access link：_[https://geocloud.cgs.gov.cn/](https://geocloud.cgs.gov.cn/) & [http://dc.ngac.org.cn/openReport](http://dc.ngac.org.cn/openReport) 
[data](data)
To make it easier for researchers from all over the world to understand the data and the paper, we have provided data cases in English and Chinese.

Chinese version:`data_example.xls`  
English version:`data_example_eng.xls`
---
### 2. Basic Processes：  

The following will introduce how to use deep learning methods in tensorflow for Chinese text classification process.
* Text pre-processing
* Aggregate words (or tokens) to form a dictionary file, the first n words can be retained
* Text to numbers, not in the dictionary file with representation
* Stage and fill the text, fill with, the text vector length unity
* Build Embedding layer
* Build the model
* Train the model, adjust the parameters to get the optimal performance of the model, get the model evaluation indexes
* Save the model and make predictions on new samples
----
### 3.Getting start
File description: the detailed function of the method can be seen in the code comments
`load_data.py `          Initialisation of configuration files and data import  
`model_train.py `        Model training  
`model_predict.py`       Model prediction
> 3.1 model_train.py  

Data encapsulation and import commands:
_from load_data import get_train_test_pd_

Pre-trained language model ALBERT import:
_from albert_zh.extract_feature import BertVector_ 

After the model was trained, the accuracy on the validation dataset was 86.40% and the F1 value was 88.46%, which gave good results.
> 3.2 model_predict.py  

`model_train.py`A folder named `model` is automatically generated when the model is trained.

The best model after training is completed is used as the predictive model: 

model_dir = './models'

Inputting the predicted Chinese text requires labelling the entity pairs in advance.  
e.g.the entity pair in this test statement is (Hubei, Hills):  

text1 = '湖北#丘陵#湖北大多数地区都属于丘陵地带。'  
_Translate to:_  
text1 = 'Hubei #Hills #Most of Hubei is hilly.'

The relationship prediction results in：`spatial` 

---
### **Contact Me：**
Email：maying2019@cug.edu.cn


