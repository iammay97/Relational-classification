# **Domain Ontology-Driven ALBERT Model for Relationship Extraction from Landslide Disaster Reports**

### **_ABSTRACT_**

Knowledge mapping provides knowledge support for geological hazards that consider factors such as complexity, suddenness, and spatial-temporal nature, and relationship extraction is a key part of knowledge mapping construction. However, unstructured data accumulated over the years cannot intuitively display complex geoscientific information, inter-information relationships, and deep connections across texts. In order to solve this problem, this study proposes a framework for the relation extraction of landslide geological disasters based on the domain ontology. Firstly, an ontology of landslide hazard chains is constructed from the collected corpus to define the conceptual architecture of landslide hazards. Then, the deep learning model is applied for the first time to the relationship extraction task, where the ALBERT (A Lite Bert) pre-trained language model is embedded to obtain character vectoring. Its textual features are fed into the BiGRU-Attention relationship extraction model for training. The resulting probabilistic weights are assigned to be summed up with the product of the states of the individual hidden layers to determine the result of the relationship classification. The comprehensive experimental results show that the ALBERT-BiGRU-Attention relationship extraction model performs best, with a Precision rate of 86.40%, a recall rate of 87.88%, and an F1 score of 88.46%. Therefore, the methodology adopted in this study can provide technical support for the construction of a knowledge map of landslide geohazards, and the visualization of the results can demonstrate the process of spatial and temporal changes of landslide hazards as well as the impact on human behavioral activities.

### _**Keywords:**_

Knowledge graph，Landslide geological disaster，Spatial and temporal characteristics，Relationship extraction，Word embedding，Deep learning model

### _Code availability section_

Name of the code:Relational-classification

Contact: maying2019@cug.edu.cn

Hardware requirements: ... 

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
