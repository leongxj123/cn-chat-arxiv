# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design](https://arxiv.org/abs/2402.05982) | Anfinsen Goes Neural (AGN) is a graphical model for conditional antibody design that combines a pre-trained protein language model with a graph neural network. It outperforms existing methods and addresses the limitation of generating unrealistic sequences. |

# 详细

[^1]: Anfinsen Goes Neural: 一种用于条件抗体设计的图模型

    Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design

    [https://arxiv.org/abs/2402.05982](https://arxiv.org/abs/2402.05982)

    Anfinsen Goes Neural (AGN) is a graphical model for conditional antibody design that combines a pre-trained protein language model with a graph neural network. It outperforms existing methods and addresses the limitation of generating unrealistic sequences.

    

    抗体设计在推动治疗学方面起着关键作用。尽管深度学习在这个领域取得了快速进展，但现有方法对一般蛋白质知识的利用有限，并假设图模型违反蛋白质的经验发现。为了解决这些限制，我们提出了Anfinsen Goes Neural (AGN)，这是一个使用预训练的蛋白质语言模型(pLM)并编码了一种关于蛋白质的重要发现，即Anfinsen's dogma的图模型。我们的框架遵循序列生成和图神经网络(GNN)进行结构预测的两步过程。实验证明，我们的方法在基准实验中优于现有方法的结果。我们还解决了非自回归模型的一个关键限制，即它们倾向于生成具有过多重复标记的不现实序列。为了解决这个问题，我们引入了基于组合的正则化项到交叉熵目标中，可以实现有效的权衡。

    Antibody design plays a pivotal role in advancing therapeutics. Although deep learning has made rapid progress in this field, existing methods make limited use of general protein knowledge and assume a graphical model (GM) that violates empirical findings on proteins. To address these limitations, we present Anfinsen Goes Neural (AGN), a graphical model that uses a pre-trained protein language model (pLM) and encodes a seminal finding on proteins called Anfinsen's dogma. Our framework follows a two-step process of sequence generation with pLM and structure prediction with graph neural network (GNN). Experiments show that our approach outperforms state-of-the-art results on benchmark experiments. We also address a critical limitation of non-autoregressive models -- namely, that they tend to generate unrealistic sequences with overly repeating tokens. To resolve this, we introduce a composition-based regularization term to the cross-entropy objective that allows an efficient trade-off be
    

