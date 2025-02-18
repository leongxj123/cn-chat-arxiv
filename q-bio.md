# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding](https://arxiv.org/abs/2402.02243) | ChatGPT在LLM规模上通过利用语言本身的收敛约束来做到超出预期的表现，但并不真正理解语义以及与感觉动作的直接联系。 |
| [^2] | [Human alignment of neural network representations.](http://arxiv.org/abs/2211.01201) | 本文研究神经网络表示与人类心理表示之间的对齐问题，发现模型规模和体系结构对对齐几乎没有影响，而训练数据集和目标函数都对对齐有很大的影响。从一个数据集中学习的神经网络表示的线性变换能显著提高对另外两个数据集中人类相似性判断的对齐性。 |

# 详细

[^1]: 语言扩展：LLMs，ChatGPT，接地，意义和理解

    Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding

    [https://arxiv.org/abs/2402.02243](https://arxiv.org/abs/2402.02243)

    ChatGPT在LLM规模上通过利用语言本身的收敛约束来做到超出预期的表现，但并不真正理解语义以及与感觉动作的直接联系。

    

    除了OpenAI可能对我们隐瞒的少量信息外，我们都大致知道ChatGPT是如何工作的（它的大型文本数据库，统计数据，向量表示以及它巨大的参数数量，其下一个词的训练等）。但我们谁也不能说我们对ChatGPT的这些资源所能做到的事情不感到惊讶。这甚至让我们有人得出结论，ChatGPT实际上理解了。它并不理解，但我们也不能说我们理解它是如何做到这一点的。我将提出关于良性偏见的一些猜想：在LLM规模上出现的收敛约束可能有助于ChatGPT做得比我们预期的好得多。这些偏见是语言本身在LLM规模上固有的，并且与ChatGPT缺乏直接的感觉动作接地以将其词与其所指的对象以及其命题与其意义联系起来密切相关。

    Apart from what (little) OpenAI may be concealing from us, we all know (roughly) how ChatGPT works (its huge text database, its statistics, its vector representations, and their huge number of parameters, its next-word training, and so on). But none of us can say (hand on heart) that we are not surprised by what ChatGPT has proved to be able to do with these resources. This has even driven some of us to conclude that ChatGPT actually understands. It is not true that it understands. But it is also not true that we understand how it can do what it can do. I will suggest some hunches about benign biases: convergent constraints that emerge at LLM scale that may be helping ChatGPT do so much better than we would have expected. These biases are inherent in the nature of language itself, at LLM scale, and they are closely linked to what it is that ChatGPT lacks, which is direct sensorimotor grounding to connect its words to their referents and its propositions to their meanings. These converg
    
[^2]: 人类对神经网络表示的对齐

    Human alignment of neural network representations. (arXiv:2211.01201v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.01201](http://arxiv.org/abs/2211.01201)

    本文研究神经网络表示与人类心理表示之间的对齐问题，发现模型规模和体系结构对对齐几乎没有影响，而训练数据集和目标函数都对对齐有很大的影响。从一个数据集中学习的神经网络表示的线性变换能显著提高对另外两个数据集中人类相似性判断的对齐性。

    

    当今的计算机视觉模型在各种视觉任务上实现了人类或接近人类水平的性能。然而，它们的体系结构、数据和学习算法与导致人类视觉的方式存在许多不同之处。本文研究影响神经网络所学习的表示与通过行为反应推断出的人类心理表示之间对齐的因素。我们发现，模型的规模和体系结构对与人类行为反应的对齐基本上没有影响，而训练数据集和目标函数则具有更大的影响。这些发现在使用两种不同任务收集的三个人类相似度判断数据集中保持一致。从一个数据集中学习的神经网络表示的线性变换显著提高了对另外两个数据集中的人类相似度判断的对齐性。此外，我们发现，一些人类概念...

    Today's computer vision models achieve human or near-human level performance across a wide variety of vision tasks. However, their architectures, data, and learning algorithms differ in numerous ways from those that give rise to human vision. In this paper, we investigate the factors that affect the alignment between the representations learned by neural networks and human mental representations inferred from behavioral responses. We find that model scale and architecture have essentially no effect on the alignment with human behavioral responses, whereas the training dataset and objective function both have a much larger impact. These findings are consistent across three datasets of human similarity judgments collected using two different tasks. Linear transformations of neural network representations learned from behavioral responses from one dataset substantially improve alignment with human similarity judgments on the other two datasets. In addition, we find that some human concept
    

