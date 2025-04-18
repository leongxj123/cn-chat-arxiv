# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation](https://arxiv.org/abs/2207.14000) | 提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。 |

# 详细

[^1]: 自然语言上的多步演绎推理：基于超领域泛化的实证研究

    Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation

    [https://arxiv.org/abs/2207.14000](https://arxiv.org/abs/2207.14000)

    提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。

    

    将深度学习与符号逻辑推理结合起来，旨在充分利用这两个领域的成功，并引起了越来越多的关注。受DeepLogic启发，该模型经过端到端训练，用于执行逻辑程序推理，我们介绍了IMA-GloVe-GA，这是一个用自然语言表达的多步推理的迭代神经推理网络。在我们的模型中，推理是使用基于RNN的迭代内存神经网络进行的，其中包含一个门关注机制。我们在PARARULES、CONCEPTRULES V1和CONCEPTRULES V2三个数据集上评估了IMA-GloVe-GA。实验结果表明，带有门关注机制的DeepLogic比DeepLogic和其他RNN基线模型能够实现更高的测试准确性。我们的模型在规则被打乱时比RoBERTa-Large实现了更好的超领域泛化性能。此外，为了解决当前多步推理数据集中推理深度不平衡的问题

    arXiv:2207.14000v2 Announce Type: replace-cross  Abstract: Combining deep learning with symbolic logic reasoning aims to capitalize on the success of both fields and is drawing increasing attention. Inspired by DeepLogic, an end-to-end model trained to perform inference on logic programs, we introduce IMA-GloVe-GA, an iterative neural inference network for multi-step reasoning expressed in natural language. In our model, reasoning is performed using an iterative memory neural network based on RNN with a gate attention mechanism. We evaluate IMA-GloVe-GA on three datasets: PARARULES, CONCEPTRULES V1 and CONCEPTRULES V2. Experimental results show DeepLogic with gate attention can achieve higher test accuracy than DeepLogic and other RNN baseline models. Our model achieves better out-of-distribution generalisation than RoBERTa-Large when the rules have been shuffled. Furthermore, to address the issue of unbalanced distribution of reasoning depths in the current multi-step reasoning datase
    

