# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs](https://rss.arxiv.org/abs/2312.05356) | 这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。 |

# 详细

[^1]: Neuron Patching: 神经元层面的模型编辑与代码生成

    Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs

    [https://rss.arxiv.org/abs/2312.05356](https://rss.arxiv.org/abs/2312.05356)

    这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。

    

    大型语言模型在软件工程中得到了成功应用，特别是在代码生成方面。更新这些模型的新知识非常昂贵，通常需要全面实现其价值。在本文中，我们提出了一种新颖有效的模型编辑方法MENT，用于在编码任务中修补LLM模型。基于生成式LLM的机制，MENT可以在预测下一个令牌时进行模型编辑，并进一步支持常见的编码任务。MENT具有高效、有效和可靠的特点。它可以通过修补1或2个神经元来纠正神经模型。作为神经元层面上生成模型编辑的先驱工作，我们规范了编辑过程并介绍了相关概念。此外，我们还引入了新的衡量方法来评估其泛化能力，并建立了一个用于进一步研究的基准。我们的方法在三个编码任务上进行了评估，包括API序列推荐、行级代码生成和伪代码到代码转换。

    Large Language Models are successfully adopted in software engineering, especially in code generation. Updating these models with new knowledge is very expensive, and is often required to fully realize their value. In this paper, we propose a novel and effective model editing approach, \textsc{MENT}, to patch LLMs in coding tasks. Based on the mechanism of generative LLMs, \textsc{MENT} enables model editing in next-token predictions, and further supports common coding tasks. \textsc{MENT} is effective, efficient, and reliable. It can correct a neural model by patching 1 or 2 neurons. As the pioneer work on neuron-level model editing of generative models, we formalize the editing process and introduce the involved concepts. Besides, we also introduce new measures to evaluate its generalization ability, and build a benchmark for further study. Our approach is evaluated on three coding tasks, including API-seq recommendation, line-level code generation, and pseudocode-to-code transaction
    

