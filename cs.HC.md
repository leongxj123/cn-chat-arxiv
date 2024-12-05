# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826) | 该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。 |
| [^2] | [OpenDriver: an open-road driver state detection dataset.](http://arxiv.org/abs/2304.04203) | OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。 |

# 详细

[^1]: 大型语言模型的预测排名

    Prediction-Powered Ranking of Large Language Models

    [https://arxiv.org/abs/2402.17826](https://arxiv.org/abs/2402.17826)

    该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。

    

    大型语言模型通常根据其与人类偏好的一致性水平进行排名--如果一个模型的输出更受人类偏好，那么它就比其他模型更好。本文提出了一种统计框架来弥合人类与模型偏好之间可能引入的不一致性。

    arXiv:2402.17826v1 Announce Type: cross  Abstract: Large language models are often ranked according to their level of alignment with human preferences -- a model is better than other models if its outputs are more frequently preferred by humans. One of the most popular ways to elicit human preferences utilizes pairwise comparisons between the outputs provided by different models to the same inputs. However, since gathering pairwise comparisons by humans is costly and time-consuming, it has become a very common practice to gather pairwise comparisons by a strong large language model -- a model strongly aligned with human preferences. Surprisingly, practitioners cannot currently measure the uncertainty that any mismatch between human and model preferences may introduce in the constructed rankings. In this work, we develop a statistical framework to bridge this gap. Given a small set of pairwise comparisons by humans and a large set of pairwise comparisons by a model, our framework provid
    
[^2]: OpenDriver: 一份开放路况驾驶员状态检测数据集

    OpenDriver: an open-road driver state detection dataset. (arXiv:2304.04203v1 [cs.AI])

    [http://arxiv.org/abs/2304.04203](http://arxiv.org/abs/2304.04203)

    OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。

    

    在现代社会中，道路安全严重依赖于驾驶员的心理和生理状态。疲劳、昏昏欲睡和压力等负面因素会影响驾驶员的反应时间和决策能力，导致交通事故的发生率增加。在众多的驾驶员行为监测研究中，可穿戴生理测量是一种实时监测驾驶员状态的方法。然而，目前在开放道路场景下，缺少驾驶员生理数据集，已有的数据集存在信号质量差、样本量小和数据收集时间短等问题。因此，本文设计并描述了一种大规模多模态驾驶数据集，用于驾驶员受损检测和生物识别数据识别。该数据集包含两种驾驶信号模态：六轴惯性信号和心电图（ECG）信号，这些信号是在100多名驾驶员遵循相同路线行驶时记录的。

    In modern society, road safety relies heavily on the psychological and physiological state of drivers. Negative factors such as fatigue, drowsiness, and stress can impair drivers' reaction time and decision making abilities, leading to an increased incidence of traffic accidents. Among the numerous studies for impaired driving detection, wearable physiological measurement is a real-time approach to monitoring a driver's state. However, currently, there are few driver physiological datasets in open road scenarios and the existing datasets suffer from issues such as poor signal quality, small sample sizes, and short data collection periods. Therefore, in this paper, a large-scale multimodal driving dataset for driver impairment detection and biometric data recognition is designed and described. The dataset contains two modalities of driving signals: six-axis inertial signals and electrocardiogram (ECG) signals, which were recorded while over one hundred drivers were following the same ro
    

