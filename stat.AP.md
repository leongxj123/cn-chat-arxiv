# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Emotion Detection with Transformers: A Comparative Study](https://arxiv.org/abs/2403.15454) | 本研究探索了在文本数据情感分类中应用基于Transformer的模型，并发现常用技术如去除标点符号和停用词可能会阻碍模型的性能，因为这些元素仍然能够传达情感或强调，而Transformer的优势在于理解文本内的语境关系。 |
| [^2] | [Extracting Emotion Phrases from Tweets using BART](https://arxiv.org/abs/2403.14050) | 本文提出了一种基于BART的情感分析方法，利用问答框架从文本中提取特定情绪短语，并通过分类器预测答案跨度位置，实现对情绪短语的精确提取。 |
| [^3] | [SportsNGEN: Sustained Generation of Multi-player Sports Gameplay](https://arxiv.org/abs/2403.12977) | SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。 |
| [^4] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |

# 详细

[^1]: 使用Transformer进行情感检测：一项比较研究

    Emotion Detection with Transformers: A Comparative Study

    [https://arxiv.org/abs/2403.15454](https://arxiv.org/abs/2403.15454)

    本研究探索了在文本数据情感分类中应用基于Transformer的模型，并发现常用技术如去除标点符号和停用词可能会阻碍模型的性能，因为这些元素仍然能够传达情感或强调，而Transformer的优势在于理解文本内的语境关系。

    

    在这项研究中，我们探讨了基于Transformer模型在文本数据情感分类中的应用。我们使用不同变体的Transformer对Emotion数据集进行训练和评估。论文还分析了一些影响模型性能的因素，比如Transformer层的微调、层的可训练性以及文本数据的预处理。我们的分析表明，常用技术如去除标点符号和停用词可能会阻碍模型的性能。这可能是因为Transformer的优势在于理解文本内的语境关系。像标点符号和停用词这样的元素仍然可以传达情感或强调，去除它们可能会破坏这种上下文。

    arXiv:2403.15454v1 Announce Type: new  Abstract: In this study, we explore the application of transformer-based models for emotion classification on text data. We train and evaluate several pre-trained transformer models, on the Emotion dataset using different variants of transformers. The paper also analyzes some factors that in-fluence the performance of the model, such as the fine-tuning of the transformer layer, the trainability of the layer, and the preprocessing of the text data. Our analysis reveals that commonly applied techniques like removing punctuation and stop words can hinder model performance. This might be because transformers strength lies in understanding contextual relationships within text. Elements like punctuation and stop words can still convey sentiment or emphasis and removing them might disrupt this context.
    
[^2]: 使用BART从推文中提取情绪短语

    Extracting Emotion Phrases from Tweets using BART

    [https://arxiv.org/abs/2403.14050](https://arxiv.org/abs/2403.14050)

    本文提出了一种基于BART的情感分析方法，利用问答框架从文本中提取特定情绪短语，并通过分类器预测答案跨度位置，实现对情绪短语的精确提取。

    

    情感分析是一项旨在识别和提取文本中情绪方面的自然语言处理任务。然而，许多现有的情感分析方法主要是对文本的整体极性进行分类，忽略了传达情绪的具体短语。在本文中，我们应用了一种基于问答框架的情感分析方法。我们利用双向自回归变换器（BART），一个预训练的序列到序列模型，从给定文本中提取放大给定情感极性的短语。我们创建一个自然语言问题，确定要提取的特定情绪，然后引导BART专注于文本中相关的情感线索。我们在BART中使用一个分类器来预测文本中答案跨度的开始和结束位置，从而帮助确定提取的情绪短语的精确边界。

    arXiv:2403.14050v2 Announce Type: replace  Abstract: Sentiment analysis is a natural language processing task that aims to identify and extract the emotional aspects of a text. However, many existing sentiment analysis methods primarily classify the overall polarity of a text, overlooking the specific phrases that convey sentiment. In this paper, we applied an approach to sentiment analysis based on a question-answering framework. Our approach leverages the power of Bidirectional Autoregressive Transformer (BART), a pre-trained sequence-to-sequence model, to extract a phrase from a given text that amplifies a given sentiment polarity. We create a natural language question that identifies the specific emotion to extract and then guide BART to pay attention to the relevant emotional cues in the text. We use a classifier within BART to predict the start and end positions of the answer span within the text, which helps to identify the precise boundaries of the extracted emotion phrase. Our
    
[^3]: SportsNGEN: 持续生成多人体育游戏

    SportsNGEN: Sustained Generation of Multi-player Sports Gameplay

    [https://arxiv.org/abs/2403.12977](https://arxiv.org/abs/2403.12977)

    SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。

    

    我们提出了一种基于Transformer解码器的模型SportsNGEN，该模型经过训练使用运动员和球追踪序列，能够生成逼真且持续的游戏场景。我们在大量专业网球追踪数据上训练和评估SportsNGEN，并展示通过将生成的模拟与射击分类器和逻辑相结合来开始和结束球赛，系统能够模拟整个网球比赛。此外，SportsNGEN的通用版本可以通过在包含该球员的比赛数据上微调来定制特定球员。我们展示了我们的模型经过良好校准，可以通过评估反事实或假设选项为教练和广播员提供洞察力。最后，我们展示了质量结果表明相同的方法适用于足球。

    arXiv:2403.12977v1 Announce Type: cross  Abstract: We present a transformer decoder based model, SportsNGEN, that is trained on sports player and ball tracking sequences that is capable of generating realistic and sustained gameplay. We train and evaluate SportsNGEN on a large database of professional tennis tracking data and demonstrate that by combining the generated simulations with a shot classifier and logic to start and end rallies, the system is capable of simulating an entire tennis match. In addition, a generic version of SportsNGEN can be customized to a specific player by fine-tuning on match data that includes that player. We show that our model is well calibrated and can be used to derive insights for coaches and broadcasters by evaluating counterfactual or what if options. Finally, we show qualitative results indicating the same approach works for football.
    
[^4]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    

