# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A ripple in time: a discontinuity in American history.](http://arxiv.org/abs/2312.01185) | 该论文通过使用向量嵌入和非线性降维方法，发现GPT-2与UMAP的结合可以提供更好的分离和聚类效果。同时，经过微调的DistilBERT模型可用于识别总统和演讲的年份。 |
| [^2] | [The Face of Populism: Examining Differences in Facial Emotional Expressions of Political Leaders Using Machine Learning.](http://arxiv.org/abs/2304.09914) | 本文使用机器学习的算法分析了来自15个不同国家的220个政治领袖的YouTube视频，总结了政治领袖面部情感表达的差异。 |

# 详细

[^1]: 时间中的涟漪：美国历史中的不连续性

    A ripple in time: a discontinuity in American history. (arXiv:2312.01185v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.01185](http://arxiv.org/abs/2312.01185)

    该论文通过使用向量嵌入和非线性降维方法，发现GPT-2与UMAP的结合可以提供更好的分离和聚类效果。同时，经过微调的DistilBERT模型可用于识别总统和演讲的年份。

    

    在这篇论文中，我们使用来自Kaggle的国情咨文数据集对美国历史的总体时间线及咨文本身的特点和性质进行了一些令人惊讶（也有些不那么令人惊讶）的观察。我们的主要方法是使用向量嵌入，如BERT（DistilBERT）和GPT-2。虽然广泛认为BERT（及其变体）最适合NLP分类任务，但我们发现GPT-2结合UMAP等非线性降维方法可以提供更好的分离和更强的聚类效果。这使得GPT-2 + UMAP成为一个有趣的替代方案。在我们的情况下，不需要对模型进行微调，预训练的GPT-2模型就足够好用。我们还使用了经过微调的DistilBERT模型来检测哪位总统发表了哪篇演讲，并取得了非常好的结果（准确率为93\% - 95\%，具体取决于运行情况）。为了确定写作年份，我们还执行了一个类似的任务。

    In this note we use the State of the Union Address (SOTU) dataset from Kaggle to make some surprising (and some not so surprising) observations pertaining to the general timeline of American history, and the character and nature of the addresses themselves. Our main approach is using vector embeddings, such as BERT (DistilBERT) and GPT-2.  While it is widely believed that BERT (and its variations) is most suitable for NLP classification tasks, we find out that GPT-2 in conjunction with nonlinear dimension reduction methods such as UMAP provide better separation and stronger clustering. This makes GPT-2 + UMAP an interesting alternative. In our case, no model fine-tuning is required, and the pre-trained out-of-the-box GPT-2 model is enough.  We also used a fine-tuned DistilBERT model for classification detecting which President delivered which address, with very good results (accuracy 93\% - 95\% depending on the run). An analogous task was performed to determine the year of writing, an
    
[^2]: 柿子政治的面孔：使用机器学习比较政治领袖面部情感表达的差异

    The Face of Populism: Examining Differences in Facial Emotional Expressions of Political Leaders Using Machine Learning. (arXiv:2304.09914v1 [cs.CY])

    [http://arxiv.org/abs/2304.09914](http://arxiv.org/abs/2304.09914)

    本文使用机器学习的算法分析了来自15个不同国家的220个政治领袖的YouTube视频，总结了政治领袖面部情感表达的差异。

    

    网络媒体已经彻底改变了政治信息在全球范围内的传播和消费方式，这种转变促使政治人物采取新的策略来捕捉和保持选民的注意力。这些策略往往依赖于情感说服和吸引。随着虚拟空间中视觉内容越来越普遍，很多政治沟通也被标志着唤起情感的视频内容和图像。本文提供了一种新的分析方法。我们将基于现有训练好的卷积神经网络架构提供的Python库fer，应用一种基于深度学习的计算机视觉算法，对描绘来自15个不同国家的政治领袖的220个YouTube视频样本进行分析。该算法返回情绪分数，每一帧都代表6种情绪状态（愤怒，厌恶，恐惧，快乐，悲伤和惊讶）和一个中性表情。

    Online media has revolutionized the way political information is disseminated and consumed on a global scale, and this shift has compelled political figures to adopt new strategies of capturing and retaining voter attention. These strategies often rely on emotional persuasion and appeal, and as visual content becomes increasingly prevalent in virtual space, much of political communication too has come to be marked by evocative video content and imagery. The present paper offers a novel approach to analyzing material of this kind. We apply a deep-learning-based computer-vision algorithm to a sample of 220 YouTube videos depicting political leaders from 15 different countries, which is based on an existing trained convolutional neural network architecture provided by the Python library fer. The algorithm returns emotion scores representing the relative presence of 6 emotional states (anger, disgust, fear, happiness, sadness, and surprise) and a neutral expression for each frame of the pr
    

