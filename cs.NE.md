# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Relational Concept Based Models.](http://arxiv.org/abs/2308.11991) | 关系概念模型是一种关系深度学习方法家族，用于在关系领域提供可解释的任务预测，相比非关系的基于概念的模型，它在泛化性能上与现有的关系模型相匹配，并支持生成量化的基于概念的解释，同时在测试时干预、超出分布情景、有限的训练数据范围和稀缺的概念监督等苛刻条件下也能有效应对。 |
| [^2] | [Learning Spiking Neural Systems with the Event-Driven Forward-Forward Process.](http://arxiv.org/abs/2303.18187) | 该论文提出了一种基于事件驱动的前向-前向和预测式前向-前向学习过程的通用化方案，用于递归电路计算每个神经元的膜电位。与依赖反馈突触调整神经电活动的尖峰神经编码不同，该模型纯在线并且时间向前，是学习带有时间尖峰信号的感觉数据模式分布表示的有前途的一种途径。 |
| [^3] | [EVOTER: Evolution of Transparent Explainable Rule-sets.](http://arxiv.org/abs/2204.10438) | EVOTER使用简单的逻辑表达式演化出透明可解释的规则集，与黑盒模型性能相似，可以揭示数据中的偏见并为未来构建可靠的AI系统提供基础。 |

# 详细

[^1]: 关于关系概念模型的研究

    Relational Concept Based Models. (arXiv:2308.11991v1 [cs.LG])

    [http://arxiv.org/abs/2308.11991](http://arxiv.org/abs/2308.11991)

    关系概念模型是一种关系深度学习方法家族，用于在关系领域提供可解释的任务预测，相比非关系的基于概念的模型，它在泛化性能上与现有的关系模型相匹配，并支持生成量化的基于概念的解释，同时在测试时干预、超出分布情景、有限的训练数据范围和稀缺的概念监督等苛刻条件下也能有效应对。

    

    在关系领域中设计可解释的深度学习模型是一个开放性挑战：可解释的深度学习方法，如基于概念的模型（CBMs），并没有设计来解决关系问题，而关系模型也没有像CBMs那样可解释。为了解决这个问题，我们提出了关系概念模型，这是一种提供可解释任务预测的关系深度学习方法家族。我们的实验从图像分类到知识图谱中的链接预测，表明关系CBMs：（i）与现有的关系黑盒的泛化性能相匹配（不同于非关系的CBMs），（ii）支持生成量化的基于概念的解释，（iii）有效应对测试时的干预，以及（iv）经受住包括超出分布情景、有限的训练数据范围和稀缺的概念监督等苛刻条件。

    The design of interpretable deep learning models working in relational domains poses an open challenge: interpretable deep learning methods, such as Concept-Based Models (CBMs), are not designed to solve relational problems, while relational models are not as interpretable as CBMs. To address this problem, we propose Relational Concept-Based Models, a family of relational deep learning methods providing interpretable task predictions. Our experiments, ranging from image classification to link prediction in knowledge graphs, show that relational CBMs (i) match generalization performance of existing relational black-boxes (as opposed to non-relational CBMs), (ii) support the generation of quantified concept-based explanations, (iii) effectively respond to test-time interventions, and (iv) withstand demanding settings including out-of-distribution scenarios, limited training data regimes, and scarce concept supervisions.
    
[^2]: 利用事件驱动的前向前向过程学习尖峰神经系统

    Learning Spiking Neural Systems with the Event-Driven Forward-Forward Process. (arXiv:2303.18187v1 [cs.NE])

    [http://arxiv.org/abs/2303.18187](http://arxiv.org/abs/2303.18187)

    该论文提出了一种基于事件驱动的前向-前向和预测式前向-前向学习过程的通用化方案，用于递归电路计算每个神经元的膜电位。与依赖反馈突触调整神经电活动的尖峰神经编码不同，该模型纯在线并且时间向前，是学习带有时间尖峰信号的感觉数据模式分布表示的有前途的一种途径。

    

    我们为使用尖峰神经元进行信息处理开发了一种新的学分分配算法，无需反馈突触。具体而言，我们提出了一种基于事件驱动的前向-前向和预测式前向-前向学习过程的通用化方案，用于迭代处理感觉输入。因此，递归电路会根据局部自下向上、自上而下和侧面的信号计算每层中每个神经元的膜电位，促进一种动态的、逐层并行的神经计算形式。与依赖反馈突触调整神经电活动的尖峰神经编码不同，我们的模型纯在线并且时间向前，这样就能够学习带有时间尖峰信号的感觉数据模式的分布表示。值得注意的是，我们在几个模式数据集上的实验结果表明，基于事件驱动的前向前向（ED-FF）框架工作正常。

    We develop a novel credit assignment algorithm for information processing with spiking neurons without requiring feedback synapses. Specifically, we propose an event-driven generalization of the forward-forward and the predictive forward-forward learning processes for a spiking neural system that iteratively processes sensory input over a stimulus window. As a result, the recurrent circuit computes the membrane potential of each neuron in each layer as a function of local bottom-up, top-down, and lateral signals, facilitating a dynamic, layer-wise parallel form of neural computation. Unlike spiking neural coding, which relies on feedback synapses to adjust neural electrical activity, our model operates purely online and forward in time, offering a promising way to learn distributed representations of sensory data patterns with temporal spike signals. Notably, our experimental results on several pattern datasets demonstrate that the even-driven forward-forward (ED-FF) framework works we
    
[^3]: EVOTER：透明可解释规则集的进化

    EVOTER: Evolution of Transparent Explainable Rule-sets. (arXiv:2204.10438v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2204.10438](http://arxiv.org/abs/2204.10438)

    EVOTER使用简单的逻辑表达式演化出透明可解释的规则集，与黑盒模型性能相似，可以揭示数据中的偏见并为未来构建可靠的AI系统提供基础。

    

    大多数AI系统是黑盒子，为给定的输入生成合理的输出。然而，某些领域具有解释能力和信任度要求，这些要求不能直接满足这些方法。因此，该论文提出了一种替代方法，即开始时模型就是透明的和可解释的。该方法使用简单的逻辑表达式演化出规则集，称为EVOTER。EVOTER在多个预测/分类和处方/政策搜索领域进行了评估，有和没有代理。结果显示，它能够发现和黑盒模型相似的有意义的规则集。这些规则可以提供领域的见解，并使数据中隐藏的偏见显性化。也可以直接对它们进行编辑，以消除偏见并添加约束。因此，EVOTER为未来构建值得信赖的AI系统的可靠基础。

    Most AI systems are black boxes generating reasonable outputs for given inputs. Some domains, however, have explainability and trustworthiness requirements that cannot be directly met by these approaches. Various methods have therefore been developed to interpret black-box models after training. This paper advocates an alternative approach where the models are transparent and explainable to begin with. This approach, EVOTER, evolves rule-sets based on simple logical expressions. The approach is evaluated in several prediction/classification and prescription/policy search domains with and without a surrogate. It is shown to discover meaningful rule sets that perform similarly to black-box models. The rules can provide insight into the domain, and make biases hidden in the data explicit. It may also be possible to edit them directly to remove biases and add constraints. EVOTER thus forms a promising foundation for building trustworthy AI systems for real-world applications in the future.
    

