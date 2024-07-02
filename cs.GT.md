# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Value of Context: Human versus Black Box Evaluators](https://arxiv.org/abs/2402.11157) | 机器学习算法是标准化的，评估所有个体时通过固定的共变量，而人类评估者通过定制共变量的获取对每个个体进行评估，我们展示在高维数据环境中，上下文的定制化优势。 |
| [^2] | [Partially Observable Stochastic Games with Neural Perception Mechanisms.](http://arxiv.org/abs/2310.11566) | 本研究提出了神经符号化部分可观测随机博弈（NS-POSGs）模型，通过融合感知机制解决了多智能体序列决策中的部分可观测性问题。其中，我们专注于一种只有部分观测信息的智能体和一种完全观测的智能体的单方面设置，并提出了一种近似计算NS-POSGs值的新方法。 |
| [^3] | [A Robust Characterization of Nash Equilibrium.](http://arxiv.org/abs/2307.03079) | 本论文通过假设在不同游戏中具有一致的行为，给出了一种鲁棒的纳什均衡特征化方法，并证明了纳什均衡是唯一满足结果主义、一致性和合理性的解概念。该结果适用于各种自然子类的游戏。 |
| [^4] | [Investigating Emergent Goal-Like Behaviour in Large Language Models Using Experimental Economics.](http://arxiv.org/abs/2305.07970) | 本研究探讨了大型语言模型的能力，发现其可以将自然语言描述转化为适当的行为，但在区分细微的合作和竞争水平方面的能力受到限制，为使用LLMs在人类决策制定背景下的伦理意义和局限性做出了贡献。 |

# 详细

[^1]: 上下文的价值：人类评估者与黑匣子评估者

    The Value of Context: Human versus Black Box Evaluators

    [https://arxiv.org/abs/2402.11157](https://arxiv.org/abs/2402.11157)

    机器学习算法是标准化的，评估所有个体时通过固定的共变量，而人类评估者通过定制共变量的获取对每个个体进行评估，我们展示在高维数据环境中，上下文的定制化优势。

    

    评估曾经只在人类专家领域内进行（例如医生进行的医学诊断），现在也可以通过机器学习算法进行。这引发了一个新的概念问题：被人类和算法评估之间有什么区别，在什么时候个人应该更喜欢其中一种形式的评估？我们提出了一个理论框架，形式化了这两种评估形式之间的一个关键区别：机器学习算法是标准化的，通过固定的共变量来评估所有个体，而人类评估者则根据个体定制获取哪些共变量。我们的框架定义并分析了这种定制化的优势——上下文的价值，在具有非常高维数据的环境中。我们表明，除非代理人对共变量的联合分布有精确的知识，更多共变量的价值超过了上下文的价值。

    arXiv:2402.11157v1 Announce Type: new  Abstract: Evaluations once solely within the domain of human experts (e.g., medical diagnosis by doctors) can now also be carried out by machine learning algorithms. This raises a new conceptual question: what is the difference between being evaluated by humans and algorithms, and when should an individual prefer one form of evaluation over the other? We propose a theoretical framework that formalizes one key distinction between the two forms of evaluation: Machine learning algorithms are standardized, fixing a common set of covariates by which to assess all individuals, while human evaluators customize which covariates are acquired to each individual. Our framework defines and analyzes the advantage of this customization -- the value of context -- in environments with very high-dimensional data. We show that unless the agent has precise knowledge about the joint distribution of covariates, the value of more covariates exceeds the value of context
    
[^2]: 具有神经感知机制的部分可观测随机博弈

    Partially Observable Stochastic Games with Neural Perception Mechanisms. (arXiv:2310.11566v1 [cs.GT])

    [http://arxiv.org/abs/2310.11566](http://arxiv.org/abs/2310.11566)

    本研究提出了神经符号化部分可观测随机博弈（NS-POSGs）模型，通过融合感知机制解决了多智能体序列决策中的部分可观测性问题。其中，我们专注于一种只有部分观测信息的智能体和一种完全观测的智能体的单方面设置，并提出了一种近似计算NS-POSGs值的新方法。

    

    随机博弈是一个为多智能体在不确定性下进行序列决策的模型。然而在现实中，智能体对环境只有部分可观测性，这使得问题在计算上具有挑战性，即使在部分可观测马尔可夫决策过程的单智能体环境中也是如此。此外，在实践中，智能体越来越多地使用基于数据的方法，例如在连续数据上训练的神经网络来感知环境。为了解决这个问题，我们提出了神经符号化部分可观测随机博弈（NS-POSGs）的模型，这是连续空间并发随机博弈的一种变体，明确地融入了感知机制。我们专注于单方面的设置，包含了一个具有离散、基于数据驱动的观测和一个具有连续观测的充分了解的智能体。我们提出了一种名为单边NS-HSVI的基于点的方法，用来近似计算单方面NS-POSGs的值，并进行了实现。

    Stochastic games are a well established model for multi-agent sequential decision making under uncertainty. In reality, though, agents have only partial observability of their environment, which makes the problem computationally challenging, even in the single-agent setting of partially observable Markov decision processes. Furthermore, in practice, agents increasingly perceive their environment using data-driven approaches such as neural networks trained on continuous data. To tackle this problem, we propose the model of neuro-symbolic partially-observable stochastic games (NS-POSGs), a variant of continuous-space concurrent stochastic games that explicitly incorporates perception mechanisms. We focus on a one-sided setting, comprising a partially-informed agent with discrete, data-driven observations and a fully-informed agent with continuous observations. We present a new point-based method, called one-sided NS-HSVI, for approximating values of one-sided NS-POSGs and implement it ba
    
[^3]: 一种鲁棒的纳什均衡特征化方法

    A Robust Characterization of Nash Equilibrium. (arXiv:2307.03079v1 [econ.TH])

    [http://arxiv.org/abs/2307.03079](http://arxiv.org/abs/2307.03079)

    本论文通过假设在不同游戏中具有一致的行为，给出了一种鲁棒的纳什均衡特征化方法，并证明了纳什均衡是唯一满足结果主义、一致性和合理性的解概念。该结果适用于各种自然子类的游戏。

    

    我们通过假设在不同游戏中具有一致的行为来给出纳什均衡的鲁棒特征化方法：纳什均衡是唯一满足结果主义、一致性和合理性的解概念。因此，每个均衡改进方法都至少违反其中一种性质。我们还证明，每个近似满足结果主义、一致性和合理性的解概念都会产生近似的纳什均衡。通过增加公理的逼近程度，后者的逼近程度可以任意提高。该结果适用于两人零和游戏、势函数游戏和图形游戏等各种自然子类的游戏。

    We give a robust characterization of Nash equilibrium by postulating coherent behavior across varying games: Nash equilibrium is the only solution concept that satisfies consequentialism, consistency, and rationality. As a consequence, every equilibrium refinement violates at least one of these properties. We moreover show that every solution concept that approximately satisfies consequentialism, consistency, and rationality returns approximate Nash equilibria. The latter approximation can be made arbitrarily good by increasing the approximation of the axioms. This result extends to various natural subclasses of games such as two-player zero-sum games, potential games, and graphical games.
    
[^4]: 利用实验经济学研究大型语言模型中出现的类似目标行为

    Investigating Emergent Goal-Like Behaviour in Large Language Models Using Experimental Economics. (arXiv:2305.07970v1 [cs.GT])

    [http://arxiv.org/abs/2305.07970](http://arxiv.org/abs/2305.07970)

    本研究探讨了大型语言模型的能力，发现其可以将自然语言描述转化为适当的行为，但在区分细微的合作和竞争水平方面的能力受到限制，为使用LLMs在人类决策制定背景下的伦理意义和局限性做出了贡献。

    

    本研究探讨了大型语言模型（LLMs），特别是GPT-3.5，实现合作、竞争、利他和自私行为的自然语言描述在社会困境下的能力。我们聚焦于迭代囚徒困境，这是一个非零和互动的经典例子，但我们的更广泛研究计划包括一系列实验经济学场景，包括最后通牒博弈、独裁者博弈和公共物品游戏。使用被试内实验设计，我们运用不同的提示信息实例化由LLM生成的智能体，表达不同的合作和竞争立场。我们评估了智能体在迭代囚徒困境中的合作水平，同时考虑到它们对合作或出尔反尔的伙伴行动的响应。我们的结果表明，LLMs在某种程度上可以将利他和自私的自然语言描述转化为适当的行为，但展示出区分合作和竞争水平的能力有限。总体而言，我们的研究为在人类决策制定的背景下使用LLMs的伦理意义和局限性提供了证据。

    In this study, we investigate the capacity of large language models (LLMs), specifically GPT-3.5, to operationalise natural language descriptions of cooperative, competitive, altruistic, and self-interested behavior in social dilemmas. Our focus is on the iterated Prisoner's Dilemma, a classic example of a non-zero-sum interaction, but our broader research program encompasses a range of experimental economics scenarios, including the ultimatum game, dictator game, and public goods game. Using a within-subject experimental design, we instantiated LLM-generated agents with various prompts that conveyed different cooperative and competitive stances. We then assessed the agents' level of cooperation in the iterated Prisoner's Dilemma, taking into account their responsiveness to the cooperative or defection actions of their partners. Our results provide evidence that LLMs can translate natural language descriptions of altruism and selfishness into appropriate behaviour to some extent, but e
    

