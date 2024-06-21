# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs](https://arxiv.org/abs/2402.11804) | 本文利用大型语言模型（LLMs）生成图形结构提示，以增强预训练的图神经网络（GNNs），提出一种新的方法论见解，实现了在任意知识图上进行低资源归纳推理的高通用性。 |
| [^2] | [A Survey of Data-Efficient Graph Learning](https://arxiv.org/abs/2402.00447) | 这项研究提出了数据高效图学习（DEGL）的概念，并总结了近期在这一领域的进展。DEGL的目标是在资源有限的场景下提高图机器学习的性能，通过探索各种最小监督方法来解决大规模标记数据的挑战。 |

# 详细

[^1]: LLM作为提示器：在任意知识图上进行低资源归纳推理

    LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs

    [https://arxiv.org/abs/2402.11804](https://arxiv.org/abs/2402.11804)

    本文利用大型语言模型（LLMs）生成图形结构提示，以增强预训练的图神经网络（GNNs），提出一种新的方法论见解，实现了在任意知识图上进行低资源归纳推理的高通用性。

    

    知识图（KG）归纳推理旨在推断训练期间未见过的新KG中缺失的事实，在各种应用中被广泛采用。KG归纳推理的一个关键挑战是处理在文本和结构方面都稀缺的低资源场景。本文尝试利用大型语言模型（LLMs）来解决这一挑战。具体来说，我们利用最先进的LLMs生成图形结构提示，以增强预训练的图神经网络（GNNs），从而为KG归纳推理方法带来新的方法论见解，以及在实践中具有很高的普适性。在方法论方面，我们引入了一种新颖的预训练和提示框架ProLINK，旨在在任意KG上进行低资源归纳推理，而无需额外训练。在实践方面，我们在36个低资源数据集上对我们的方法进行了实验评估。

    arXiv:2402.11804v1 Announce Type: new  Abstract: Knowledge Graph (KG) inductive reasoning, which aims to infer missing facts from new KGs that are not seen during training, has been widely adopted in various applications. One critical challenge of KG inductive reasoning is handling low-resource scenarios with scarcity in both textual and structural aspects. In this paper, we attempt to address this challenge with Large Language Models (LLMs). Particularly, we utilize the state-of-the-art LLMs to generate a graph-structural prompt to enhance the pre-trained Graph Neural Networks (GNNs), which brings us new methodological insights into the KG inductive reasoning methods, as well as high generalizability in practice. On the methodological side, we introduce a novel pretraining and prompting framework ProLINK, designed for low-resource inductive reasoning across arbitrary KGs without requiring additional training. On the practical side, we experimentally evaluate our approach on 36 low-res
    
[^2]: 数据高效图学习的综述

    A Survey of Data-Efficient Graph Learning

    [https://arxiv.org/abs/2402.00447](https://arxiv.org/abs/2402.00447)

    这项研究提出了数据高效图学习（DEGL）的概念，并总结了近期在这一领域的进展。DEGL的目标是在资源有限的场景下提高图机器学习的性能，通过探索各种最小监督方法来解决大规模标记数据的挑战。

    

    图结构化数据在社交网络到生物化学分析等领域中广泛存在，是各种现实世界系统的基础。虽然图神经网络在建模这种数据方面表现出色，但它们的成功往往依赖于大量标记数据，这在标注资源有限的实际场景中构成了挑战。为了解决这个问题，我们致力于通过探索各种最小监督方法来提高低资源设置下的图机器学习性能。本文介绍了一种新颖的数据高效图学习(DEGL)的研究前沿，并提供了对DEGL当前进展的首次综述。我们首先强调了使用大规模标记数据训练模型所固有的挑战，为我们对DEGL的探索铺平了道路。接下来，我们从几个关键方面系统地回顾了这一主题的最新进展，其中包括...

    Graph-structured data, prevalent in domains ranging from social networks to biochemical analysis, serve as the foundation for diverse real-world systems. While graph neural networks demonstrate proficiency in modeling this type of data, their success is often reliant on significant amounts of labeled data, posing a challenge in practical scenarios with limited annotation resources. To tackle this problem, tremendous efforts have been devoted to enhancing graph machine learning performance under low-resource settings by exploring various approaches to minimal supervision. In this paper, we introduce a novel concept of Data-Efficient Graph Learning (DEGL) as a research frontier, and present the first survey that summarizes the current progress of DEGL. We initiate by highlighting the challenges inherent in training models with large labeled data, paving the way for our exploration into DEGL. Next, we systematically review recent advances on this topic from several key aspects, including 
    

