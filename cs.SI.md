# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Microstructures and Accuracy of Graph Recall by Large Language Models](https://arxiv.org/abs/2402.11821) | 本研究首次系统研究了大型语言模型对图形召回的准确性和偏见微结构，探讨了它们与人类的异同以及对其他图形推理任务的影响。 |
| [^2] | [Latent Graph Diffusion: A Unified Framework for Generation and Prediction on Graphs](https://arxiv.org/abs/2402.02518) | 本文提出了一种统一框架，能够使用一个模型解决各级别和各类型的图学习任务，通过潜在图扩散模型生成和预测节点、边和图级别的特征，具有可证明的保证，并在实验证明了其有效性。 |

# 详细

[^1]: 大型语言模型对图形召回的微结构和准确性

    Microstructures and Accuracy of Graph Recall by Large Language Models

    [https://arxiv.org/abs/2402.11821](https://arxiv.org/abs/2402.11821)

    本研究首次系统研究了大型语言模型对图形召回的准确性和偏见微结构，探讨了它们与人类的异同以及对其他图形推理任务的影响。

    

    图形数据对许多应用至关重要，其中很多数据以文本格式描述关系。因此，准确地召回和编码先前文本中描述的图形是大型语言模型(LLMs)需要展示的基本但关键能力，以执行涉及图形结构信息的推理任务。人类在图形召回方面的表现已被认知科学家研究了几十年，发现其经常呈现与人类处理社会关系一致的某些结构性偏见模式。然而，迄今为止，我们很少了解LLMs在类似图形召回任务中的行为：它们召回的图形是否也呈现某些偏见模式，如果是，它们与人类的表现有何不同并如何影响其他图形推理任务？在这项研究中，我们进行了第一次对LLMs进行图形召回的系统研究，研究其准确性和偏见微结构（局部结构）。

    arXiv:2402.11821v1 Announce Type: cross  Abstract: Graphs data is crucial for many applications, and much of it exists in the relations described in textual format. As a result, being able to accurately recall and encode a graph described in earlier text is a basic yet pivotal ability that LLMs need to demonstrate if they are to perform reasoning tasks that involve graph-structured information. Human performance at graph recall by has been studied by cognitive scientists for decades, and has been found to often exhibit certain structural patterns of bias that align with human handling of social relationships. To date, however, we know little about how LLMs behave in analogous graph recall tasks: do their recalled graphs also exhibit certain biased patterns, and if so, how do they compare with humans and affect other graph reasoning tasks? In this work, we perform the first systematical study of graph recall by LLMs, investigating the accuracy and biased microstructures (local structura
    
[^2]: 潜在图扩散：一种在图上生成和预测的统一框架

    Latent Graph Diffusion: A Unified Framework for Generation and Prediction on Graphs

    [https://arxiv.org/abs/2402.02518](https://arxiv.org/abs/2402.02518)

    本文提出了一种统一框架，能够使用一个模型解决各级别和各类型的图学习任务，通过潜在图扩散模型生成和预测节点、边和图级别的特征，具有可证明的保证，并在实验证明了其有效性。

    

    本文提出了第一个框架，可以使用一个模型解决各级别（节点、边和图）和各类型（生成、回归和分类）的图学习任务。首先，我们提出了潜在图扩散（LGD），一种能够同时生成节点、边和图级别特征的生成模型。通过将图结构和特征嵌入潜在空间，利用强大的编码器进行解码，然后在潜在空间中训练扩散模型，我们实现了这个目标。LGD还可以通过特殊设计的交叉注意力机制进行条件生成。然后，我们将回归和分类等预测任务形式化为（条件）生成，这使得我们的LGD能够通过可证明的保证来解决各级别和各类型的任务。通过大量的实验证明了我们框架的有效性，其中我们的模型在各项指标上取得了最先进或高度竞争力的结果。

    In this paper, we propose the first framework that enables solving graph learning tasks of all levels (node, edge and graph) and all types (generation, regression and classification) with one model. We first propose Latent Graph Diffusion (LGD), a generative model that can generate node, edge, and graph-level features of all categories simultaneously. We achieve this goal by embedding the graph structures and features into a latent space leveraging a powerful encoder which can also be decoded, then training a diffusion model in the latent space. LGD is also capable of conditional generation through a specifically designed cross-attention mechanism. Then we formulate prediction tasks including regression and classification as (conditional) generation, which enables our LGD to solve tasks of all levels and all types with provable guarantees. We verify the effectiveness of our framework with extensive experiments, where our models achieve state-of-the-art or highly competitive results acr
    

