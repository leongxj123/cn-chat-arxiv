# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention, Compilation, and Solver-based Symbolic Analysis are All You Need.](http://arxiv.org/abs/2306.06755) | 本文提出了一种基于大型语言模型的代码相互转换方法，利用注意力机制、编译和符号执行测试生成进行等价测试。在广泛的实验中，表明该方法在编译和运行时等价准确性等方面优于其他转换器和翻译工具。 |

# 详细

[^1]: 注意力、编译和基于求解器的符号分析是您所需要的一切

    Attention, Compilation, and Solver-based Symbolic Analysis are All You Need. (arXiv:2306.06755v2 [cs.PL] UPDATED)

    [http://arxiv.org/abs/2306.06755](http://arxiv.org/abs/2306.06755)

    本文提出了一种基于大型语言模型的代码相互转换方法，利用注意力机制、编译和符号执行测试生成进行等价测试。在广泛的实验中，表明该方法在编译和运行时等价准确性等方面优于其他转换器和翻译工具。

    

    在本文中，我们提出了一种基于大型语言模型的Java到Python（J2P）和Python到Java（P2J）代码相互转换方法，并介绍了一个名为CoTran的相关工具。我们的方法利用了大型语言模型的注意力机制、编译和基于符号执行的测试生成，用于输入和输出程序之间的等价测试。具体而言，我们修改了典型的大型语言模型训练循环，加入了编译器和符号执行损失。通过在超过57,000个Java-Python等价对的基准测试中将CoTran与其他12个转换器和基于大型语言模型的翻译工具进行广泛的实验比较，我们发现CoTran在诸如编译和运行时等价准确性等相关指标上表现优于它们。例如，我们的工具在J2P转换中获得97.43%的编译准确性和49.66%的运行时等价准确性，而最接近的竞争工具分别只有92.84%和40.95%。

    In this paper, we present a Java-to-Python (J2P) and Python-to-Java (P2J) back-to-back code translation method, and an associated tool called CoTran, based on large language models (LLMs). Our method leverages the attention mechanism of LLMs, compilation, and symbolic execution-based test generation for equivalence testing between the input and output programs. More precisely, we modify the typical LLM training loop to incorporate compiler and symbolic execution loss. Via extensive experiments comparing CoTran with 12 other transpilers and LLM-based translation tools over a benchmark of more than 57,000 Java-Python equivalent pairs, we show that CoTran outperforms them on relevant metrics such as compilation and runtime equivalence accuracy. For example, our tool gets 97.43% compilation accuracy and 49.66% runtime equivalence accuracy for J2P translation, whereas the nearest competing tool only gets 92.84% and 40.95% respectively.
    

