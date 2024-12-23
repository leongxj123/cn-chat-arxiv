# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Detecting Throat Cancer from Speech Signals Using Machine Learning: A Reproducible Literature Review.](http://arxiv.org/abs/2307.09230) | 本研究对使用机器学习和人工智能从语音记录中检测喉癌的文献进行了综述，发现了22篇相关论文，讨论了它们的方法和结果。研究使用了神经网络和梅尔频率倒谱系数提取音频特征，并通过迁移学习实现了分类，取得了一定的准确率。 |

# 详细

[^1]: 使用机器学习从语音信号中检测喉癌：可复现的文献综述

    Detecting Throat Cancer from Speech Signals Using Machine Learning: A Reproducible Literature Review. (arXiv:2307.09230v1 [cs.LG])

    [http://arxiv.org/abs/2307.09230](http://arxiv.org/abs/2307.09230)

    本研究对使用机器学习和人工智能从语音记录中检测喉癌的文献进行了综述，发现了22篇相关论文，讨论了它们的方法和结果。研究使用了神经网络和梅尔频率倒谱系数提取音频特征，并通过迁移学习实现了分类，取得了一定的准确率。

    

    本文对使用机器学习和人工智能从语音记录中检测喉癌的当前文献进行了范围评估。我们找到了22篇相关论文，并讨论了它们的方法和结果。我们将这些论文分为两组 - 九篇进行二分类，13篇进行多类别分类。这些论文提出了一系列方法，其中最常见的是使用神经网络。在分类之前还从音频中提取了许多特征，其中最常见的是梅尔频率倒谱系数。在这次搜索中未找到任何带有代码库的论文，因此无法复现。因此，我们创建了一个公开可用的代码库来训练自己的分类器。我们在一个多类别问题上使用迁移学习，将三种病理和健康对照进行分类。使用这种技术，我们取得了53.54%的加权平均召回率、83.14%的敏感性和特异性。

    In this work we perform a scoping review of the current literature on the detection of throat cancer from speech recordings using machine learning and artificial intelligence. We find 22 papers within this area and discuss their methods and results. We split these papers into two groups - nine performing binary classification, and 13 performing multi-class classification. The papers present a range of methods with neural networks being most commonly implemented. Many features are also extracted from the audio before classification, with the most common bring mel-frequency cepstral coefficients. None of the papers found in this search have associated code repositories and as such are not reproducible. Therefore, we create a publicly available code repository of our own classifiers. We use transfer learning on a multi-class problem, classifying three pathologies and healthy controls. Using this technique we achieve an unweighted average recall of 53.54%, sensitivity of 83.14%, and specif
    

