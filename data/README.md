
# Experimental data pre-processing instructions document
_The data refers to Chinese text data, we only provide 100 Chinese data for running, if you need all the data, please contact us.

Chinese version:`data_example.xls`

English version:`data_example_eng.xls`

----
> 1. relation_bar_chart.py  
The file generates a bar chart visualisation of the list data in order to facilitate statistical data.

**The data input is：**`data_example.xls`

**The output result is:**  `bar_chart.png`

----
> 2. data_into_train_test.py  
The file in order to divide the data into a training dataset and a test dataset.

**The data input is：** `data_example.xls`

**The output result is:**

total       100<br>
spatial      29<br>
property     19<br>
follow       10<br>
condition     9<br>
function      9<br>
casual        8<br>
temporal      8<br>
unknown       5 <br>
Name: relationship, dtype: int64<br>
----
> 3. load_data.py  
The files are for labelling relationships, calculating the maximum length, average length, minimum length, etc. of the text.。

**Label classification results for:**  
`{
  "property": 1,
  "spatial": 2,
  "function": 3,
  "casual": 4,
  "unknown": 5,
  "temporal": 6,
  "condition": 7,
  "follow": 8
}`

**The data input is：** `data_example.xls`

**The output result is:**  

|  | text_len   |
|------|------------|
|count| 80.000000  |
|mean| 79.750000  |
|min| 20.000000 |
|25%| 55.750000  |
|50%| 74.000000  |
|75%| 97.500000 |
|max| 173.000000 |

----

### Contact Me  

email：maying2019@cug.edu.cn