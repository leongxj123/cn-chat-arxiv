# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Techniques for Measuring the Inferential Strength of Forgetting Policies](https://arxiv.org/abs/2404.02454) | 本文提出了一种衡量原理论推理强度变化的损失函数，并使用Problog工具计算损失度量，最终得出了关于不同遗忘策略强度的研究方法和实际应用示例。 |
| [^2] | [Lifted Inference beyond First-Order Logic.](http://arxiv.org/abs/2308.11738) | 这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。 |

# 详细

[^1]: 衡量遗忘策略的推理强度技术

    Techniques for Measuring the Inferential Strength of Forgetting Policies

    [https://arxiv.org/abs/2404.02454](https://arxiv.org/abs/2404.02454)

    本文提出了一种衡量原理论推理强度变化的损失函数，并使用Problog工具计算损失度量，最终得出了关于不同遗忘策略强度的研究方法和实际应用示例。

    

    知识表示中的遗忘技术被证明是一种强大且有广泛应用的知识工程工具。然而，关于不同的遗忘策略或不同遗忘操作符的使用如何影响原理论的推理强度几乎没有研究。本文旨在根据模型计数和概率理论的直觉定义用于衡量推理强度变化的损失函数。研究了此类损失度量的性质，并提出了一种实用的知识工程工具，用于使用Problog计算损失度量。论文包括一个用于研究和确定不同遗忘策略强度的工作方法，以及展示如何利用Problog应用理论结果的具体示例。虽然重点是遗忘，但结果更为普遍，并且应具有更广泛的应用。

    arXiv:2404.02454v1 Announce Type: new  Abstract: The technique of forgetting in knowledge representation has been shown to be a powerful and useful knowledge engineering tool with widespread application. Yet, very little research has been done on how different policies of forgetting, or use of different forgetting operators, affects the inferential strength of the original theory. The goal of this paper is to define loss functions for measuring changes in inferential strength based on intuitions from model counting and probability theory. Properties of such loss measures are studied and a pragmatic knowledge engineering tool is proposed for computing loss measures using Problog. The paper includes a working methodology for studying and determining the strength of different forgetting policies, in addition to concrete examples showing how to apply the theoretical results using Problog. Although the focus is on forgetting, the results are much more general and should have wider applicati
    
[^2]: 超出一阶逻辑的提升推理

    Lifted Inference beyond First-Order Logic. (arXiv:2308.11738v1 [cs.AI])

    [http://arxiv.org/abs/2308.11738](http://arxiv.org/abs/2308.11738)

    这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。

    

    在统计关系学习模型中，加权一阶模型计数(WFOMC)是概率推理的基础。由于WFOMC在一般情况下是不可计算的（$\#$P完全），因此能够在多项式时间内进行WFOMC的逻辑碎片非常有意义。这样的碎片被称为域可提升。最近的研究表明，在计数量词（$\mathrm{C^2}$）扩展的两个变量的一阶逻辑片段中，可以进行域提升。然而，许多真实世界数据的属性，如引用网络中的非循环性和社交网络中的连通性，不能在$\mathrm{C^2}$或一阶逻辑中建模。在这项工作中，我们扩展了$\mathrm{C^2}$的域可提升性，包括多个这样的属性。我们证明了在将$\mathrm{C^2}$句子的一个关系限定为表示有向无环图、连通图、树（或有向树）或森林（或有向森林）时，它仍然保持了域可提升性。所有我们的结果都是...

    Weighted First Order Model Counting (WFOMC) is fundamental to probabilistic inference in statistical relational learning models. As WFOMC is known to be intractable in general ($\#$P-complete), logical fragments that admit polynomial time WFOMC are of significant interest. Such fragments are called domain liftable. Recent works have shown that the two-variable fragment of first order logic extended with counting quantifiers ($\mathrm{C^2}$) is domain-liftable. However, many properties of real-world data, like acyclicity in citation networks and connectivity in social networks, cannot be modeled in $\mathrm{C^2}$, or first order logic in general. In this work, we expand the domain liftability of $\mathrm{C^2}$ with multiple such properties. We show that any $\mathrm{C^2}$ sentence remains domain liftable when one of its relations is restricted to represent a directed acyclic graph, a connected graph, a tree (resp. a directed tree) or a forest (resp. a directed forest). All our results r
    

