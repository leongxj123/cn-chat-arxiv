# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evidence of Meaning in Language Models Trained on Programs.](http://arxiv.org/abs/2305.11169) | 该论文证明了，通过在程序语料库上训练语言模型，即使没有针对学习语言语义提供归纳偏差，语言模型仍然能够学习含义。线性探测器能够从模型状态中提取程序状态的抽象，准确性与模型泛化到新程序的能力显著相关。 |

# 详细

[^1]: 在编程语言模型中发现语义的证据

    Evidence of Meaning in Language Models Trained on Programs. (arXiv:2305.11169v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.11169](http://arxiv.org/abs/2305.11169)

    该论文证明了，通过在程序语料库上训练语言模型，即使没有针对学习语言语义提供归纳偏差，语言模型仍然能够学习含义。线性探测器能够从模型状态中提取程序状态的抽象，准确性与模型泛化到新程序的能力显著相关。

    

    我们提供证据表明，尽管被训练只是执行文本上的下一个标记预测，特别是一个程序语料库，语言模型仍然能够学习含义。每个程序都以（文本）输入输出示例的形式作为规范。与程序一起工作使我们能够精确地定义与语言中有关含义的概念（例如，正确性和语义），使得程序综合成为一个中间测试平台，用于表征语言模型中是否存在含义的存在（或不存在）。我们首先在程序语料库上训练了一个Transformer模型，然后探查了已经完成规范的程序时，经过训练的模型的隐藏状态。尽管没有针对学习语言语义提供归纳偏差，但我们发现，线性探测器能够从模型状态中提取当前和未来程序状态的抽象。此外，线性探测器的准确性与模型泛化到新程序的能力强有力、统计学显著地相关。

    We present evidence that language models can learn meaning despite being trained only to perform next token prediction on text, specifically a corpus of programs. Each program is preceded by a specification in the form of (textual) input-output examples. Working with programs enables us to precisely define concepts relevant to meaning in language (e.g., correctness and semantics), making program synthesis well-suited as an intermediate testbed for characterizing the presence (or absence) of meaning in language models.  We first train a Transformer model on the corpus of programs, then probe the trained model's hidden states as it completes a program given a specification. Despite providing no inductive bias toward learning the semantics of the language, we find that a linear probe is able to extract abstractions of both current and future program states from the model states. Moreover, there is a strong, statistically significant correlation between the accuracy of the probe and the mo
    

