# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TREET: TRansfer Entropy Estimation via Transformer](https://arxiv.org/abs/2402.06919) | 本研究提出了TREET，一种基于Transformer的传输熵估计方法，通过引入Donsker-Vardhan表示法和注意力机制，实现了对稳定过程的传输熵估计。我们设计了估计TE的优化方案，并展示了通过联合优化方案优化通信通道容量和估计器的记忆能力。 |
| [^2] | [Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains](https://arxiv.org/abs/2402.04161) | 提出了一个新的框架，通过马尔可夫链的视角研究了注意力模型的顺序建模能力，理论上刻画了单层Transformer的损失景观并发现了全局最小值和坏局部最小值的存在。 |
| [^3] | [Limits to Reservoir Learning.](http://arxiv.org/abs/2307.14474) | 这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。 |

# 详细

[^1]: TREET: 基于Transformer的传输熵估计

    TREET: TRansfer Entropy Estimation via Transformer

    [https://arxiv.org/abs/2402.06919](https://arxiv.org/abs/2402.06919)

    本研究提出了TREET，一种基于Transformer的传输熵估计方法，通过引入Donsker-Vardhan表示法和注意力机制，实现了对稳定过程的传输熵估计。我们设计了估计TE的优化方案，并展示了通过联合优化方案优化通信通道容量和估计器的记忆能力。

    

    传输熵（TE）是信息论中揭示过程之间信息流动方向的度量，对各种实际应用提供了宝贵的见解。本研究提出了一种名为TREET的基于Transformer的传输熵估计方法，用于估计稳定过程的TE。所提出的方法利用Donsker-Vardhan（DV）表示法对TE进行估计，并利用注意力机制进行神经估计任务。我们对TREET进行了详细的理论和实证研究，并将其与现有方法进行了比较。为了增加其适用性，我们设计了一种基于功能表示引理的估计TE优化方案。之后，我们利用联合优化方案来优化具有记忆性的通信通道容量，这是信息论中的一个典型优化问题，并展示了我们估计器的记忆能力。

    Transfer entropy (TE) is a measurement in information theory that reveals the directional flow of information between processes, providing valuable insights for a wide range of real-world applications. This work proposes Transfer Entropy Estimation via Transformers (TREET), a novel transformer-based approach for estimating the TE for stationary processes. The proposed approach employs Donsker-Vardhan (DV) representation to TE and leverages the attention mechanism for the task of neural estimation. We propose a detailed theoretical and empirical study of the TREET, comparing it to existing methods. To increase its applicability, we design an estimated TE optimization scheme that is motivated by the functional representation lemma. Afterwards, we take advantage of the joint optimization scheme to optimize the capacity of communication channels with memory, which is a canonical optimization problem in information theory, and show the memory capabilities of our estimator. Finally, we apply
    
[^2]: 基于马尔可夫链的注意力模型的规范分析框架：通过马尔可夫链研究Transformer的顺序建模能力

    Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains

    [https://arxiv.org/abs/2402.04161](https://arxiv.org/abs/2402.04161)

    提出了一个新的框架，通过马尔可夫链的视角研究了注意力模型的顺序建模能力，理论上刻画了单层Transformer的损失景观并发现了全局最小值和坏局部最小值的存在。

    

    近年来，基于注意力的Transformer在包括自然语言在内的多个领域取得了巨大成功。其中一个关键因素是生成式预训练过程，模型在此过程中通过自回归的方式在大型文本语料库上进行训练。为了揭示这一现象，我们提出了一个新的框架，通过马尔可夫链的视角，允许理论和系统实验来研究Transformer的顺序建模能力。受到自然语言的马尔可夫性质的启发，我们将数据建模为一个马尔可夫源，并利用这个框架系统地研究数据分布特性、Transformer架构、学到的分布和最终模型性能之间的相互作用。特别地，我们理论上刻画了单层Transformer的损失景观，并展示了全局最小值和坏局部最小值的存在，这取决于具体的数据性质。

    In recent years, attention-based transformers have achieved tremendous success across a variety of disciplines including natural languages. A key ingredient behind their success is the generative pretraining procedure, during which these models are trained on a large text corpus in an auto-regressive manner. To shed light on this phenomenon, we propose a new framework that allows both theory and systematic experiments to study the sequential modeling capabilities of transformers through the lens of Markov chains. Inspired by the Markovianity of natural languages, we model the data as a Markovian source and utilize this framework to systematically study the interplay between the data-distributional properties, the transformer architecture, the learnt distribution, and the final model performance. In particular, we theoretically characterize the loss landscape of single-layer transformers and show the existence of global minima and bad local minima contingent upon the specific data chara
    
[^3]: 河川学习的限制。

    Limits to Reservoir Learning. (arXiv:2307.14474v1 [cs.LG])

    [http://arxiv.org/abs/2307.14474](http://arxiv.org/abs/2307.14474)

    这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。

    

    在这项工作中，我们根据物理学所暗示的计算限制来限制机器学习的能力。我们首先考虑信息处理能力（IPC），这是一个对信号集合到完整函数基的期望平方误差进行归一化的指标。我们使用IPC来衡量噪声下储水库计算机（一种特殊的循环网络）的性能降低。首先，我们证明IPC在系统尺寸n上是一个多项式，即使考虑到n个输出信号的$2^n$个可能的逐点乘积。接下来，我们认为这种退化意味着在储水库噪声存在的情况下，储水库所表示的函数族需要指数数量的样本来进行学习。最后，我们讨论了在没有噪声的情况下，同一集合的$2^n$个函数在进行二元分类时的性能。

    In this work, we bound a machine's ability to learn based on computational limitations implied by physicality. We start by considering the information processing capacity (IPC), a normalized measure of the expected squared error of a collection of signals to a complete basis of functions. We use the IPC to measure the degradation under noise of the performance of reservoir computers, a particular kind of recurrent network, when constrained by physical considerations. First, we show that the IPC is at most a polynomial in the system size $n$, even when considering the collection of $2^n$ possible pointwise products of the $n$ output signals. Next, we argue that this degradation implies that the family of functions represented by the reservoir requires an exponential number of samples to learn in the presence of the reservoir's noise. Finally, we conclude with a discussion of the performance of the same collection of $2^n$ functions without noise when being used for binary classification
    

