# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks.](http://arxiv.org/abs/2305.01626) | 该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。 |

# 详细

[^1]: 基于语音的基础语法：自发联接的自监督深度神经网络

    Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks. (arXiv:2305.01626v1 [cs.CL])

    [http://arxiv.org/abs/2305.01626](http://arxiv.org/abs/2305.01626)

    该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。

    

    语法的计算模型主要基于文本。本文提出了一种完全无监督的方法，可以直接从原始语音中建立基础语法模型。我们重点研究了最普遍和基本的语法特性之一——联接。我们介绍了自发联接现象：卷积神经网络(CNN)在个别单词的声学记录上训练时，开始产生输出，这些输出将两个甚至三个单词连接在一起，而不会接触到具有多个单词的输入数据。此外，训练两个单词的网络可以学习将单词嵌入到新的未见过的单词组合中。据我们所知，这是在生成对抗网络环境下训练的原始语音CNN以前未报道的属性，它不仅对我们理解这些体系结构的学习方式有影响，还对建立从原始声学输入中的语法及其演化的模型有影响。

    Computational models of syntax are predominantly text-based. Here we propose that basic syntax can be modeled directly from raw speech in a fully unsupervised way. We focus on one of the most ubiquitous and basic properties of syntax -- concatenation. We introduce spontaneous concatenation: a phenomenon where convolutional neural networks (CNNs) trained on acoustic recordings of individual words start generating outputs with two or even three words concatenated without ever accessing data with multiple words in the input. Additionally, networks trained on two words learn to embed words into novel unobserved word combinations. To our knowledge, this is a previously unreported property of CNNs trained on raw speech in the Generative Adversarial Network setting and has implications both for our understanding of how these architectures learn as well as for modeling syntax and its evolution from raw acoustic inputs.
    

