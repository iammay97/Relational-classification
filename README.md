&emsp;&emsp;运行该项目的模型训练和模型预测脚本需要准备BERT中文版的模型数据，下载网址为：[https://github.com/google-research/bert/blob/master/multilingual.md](https://github.com/google-research/bert/blob/master/multilingual.md) 。

&emsp;&emsp;利用笔者自己收集的8000个样本，对关系抽取进行尝试。人物关系共分为8类，如下：

```json
{
  "空间": 0,
  "属性": 1,
  "因果": 2,
  "功能": 3,
  "条件": 4,
  "跟随": 5,
  "时间": 6,
  "unkonwn": 7,
}
```

&emsp;&emsp;关系类别频数分布条形图如下：

![](https://github.com/percent4/people_relation_extract/blob/master/data/bar_chart.png)

&emsp;&emsp;模型结构： BERT + 双向GRU + Attention + FC 

![](https://github.com/percent4/people_relation_extract/blob/master/model.png)
