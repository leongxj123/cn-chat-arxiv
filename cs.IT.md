# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Large Language Models Learn Independent Causal Mechanisms?](https://arxiv.org/abs/2402.02636) | 本论文研究在大型语言模型中学习独立因果机制的方法，以增强模型在分布变化下的鲁棒性和泛化能力。 |

# 详细

[^1]: 大型语言模型能否学习独立的因果机制？

    Can Large Language Models Learn Independent Causal Mechanisms?

    [https://arxiv.org/abs/2402.02636](https://arxiv.org/abs/2402.02636)

    本论文研究在大型语言模型中学习独立因果机制的方法，以增强模型在分布变化下的鲁棒性和泛化能力。

    

    尽管大型语言模型（LLMs）在语言建模和复杂推理任务中表现出色，但在不常见的环境设置或分布变化的任务中，LLMs的泛化能力仍然不足。目前通常通过增加训练数据来缓解这个问题。然而，这种方法是脆弱的，因为任务的范围可能无法预测或可能会发生变化，并且使用新数据更新模型通常需要大量的额外训练。相反，那些学习抽象变量和因果关系的系统，如因果模型，可以表现出对分布变化的更强稳健性。其中一个原因是存在并使用独立因果机制（ICMs），表示只稀疏交互的高层概念。在这项工作中，我们应用因果性的两个概念，在LLMs中学习ICMs。我们开发了一个由多个稀疏交互的语言模型组成的新LLM架构。

    Despite impressive performance on language modelling and complex reasoning tasks, Large Language Models (LLMs) fall short on the same tasks in uncommon settings or with distribution shifts, exhibiting some lack of generalisation ability. This issue has usually been alleviated by feeding more training data into the LLM. However, this method is brittle, as the scope of tasks may not be readily predictable or may evolve, and updating the model with new data generally requires extensive additional training. By contrast, systems, such as causal models, that learn abstract variables and causal relationships can demonstrate increased robustness against changes in the distribution. One reason for this success is the existence and use of Independent Causal Mechanisms (ICMs) representing high-level concepts that only sparsely interact. In this work, we apply two concepts from causality to learn ICMs within LLMs. We develop a new LLM architecture composed of multiple sparsely interacting language
    

