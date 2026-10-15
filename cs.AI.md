# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Teaching MLP More Graph Information: A Three-stage Multitask Knowledge Distillation Framework](https://arxiv.org/abs/2403.01079) | 提出了一个新的三阶段多任务知识蒸馏框架，使用位置编码来捕捉位置信息，引入神经热核处理图数据，通过隐藏层输出匹配提高学生多层感知机的性能。 |
| [^2] | [ServeFlow: A Fast-Slow Model Architecture for Network Traffic Analysis](https://arxiv.org/abs/2402.03694) | ServeFlow提出了一种快速-慢速模型架构，用于网络流量分析。通过精心选择收集数据包的数量和应用于不同流量的模型，ServeFlow实现了最小延迟、高服务率和高准确性之间的平衡。在测试中，ServeFlow能够在16ms内对76.3%的流量进行推理，这是中位数推理时间的40.5倍加速！ |
| [^3] | [Decidable Reasoning About Time in Finite-Domain Situation Calculus Theories](https://arxiv.org/abs/2402.03164) | 本论文研究了有限域情境演算理论中关于时间的可判定推理。通过引入具有受限状态公理和比较操作符的钟表作为实值fluents，解决了检查可达情境是否满足给定公式的问题的不可判定性，并给出了一个可判定的过程来确定行动序列。 |
| [^4] | [LLMRec: Benchmarking Large Language Models on Recommendation Task.](http://arxiv.org/abs/2308.12241) | LLMRec是一个基于LLMs的推荐系统，用于对LLMs在推荐任务上进行基准测试。研究结果显示，LLMs在顺序推荐和直接推荐等准确性任务方面表现出中等熟练程度，并且在指令遵循能力方面具有与最先进方法相当的性能。 |
| [^5] | [NeuSE: Neural SE(3)-Equivariant Embedding for Consistent Spatial Understanding with Objects.](http://arxiv.org/abs/2303.07308) | NeuSE是一种用于对象的神经网络嵌入算法，支持实现与长期场景变化一致的对象空间理解。它可以编码完整的形状信息，并与物理世界中的对象同时变换，能够直接推断相对帧变换和相机姿态约束，并维持适应变化的轻量级对象地图。 |

# 详细

[^1]: 教授多层感知机更多图信息：三阶段多任务知识蒸馏框架

    Teaching MLP More Graph Information: A Three-stage Multitask Knowledge Distillation Framework

    [https://arxiv.org/abs/2403.01079](https://arxiv.org/abs/2403.01079)

    提出了一个新的三阶段多任务知识蒸馏框架，使用位置编码来捕捉位置信息，引入神经热核处理图数据，通过隐藏层输出匹配提高学生多层感知机的性能。

    

    我们研究了图神经网络在大规模图数据集上进行推理任务时面临的挑战：巨大的时间和内存消耗，并尝试通过减少对图结构的依赖来克服这一问题。尽管将图知识蒸馏到学生多层感知机是一个不错的想法，但它面临两个主要问题：位置信息丢失和泛化能力低。为了解决这些问题，我们提出了一种新的三阶段多任务蒸馏框架。具体地，我们使用位置编码来捕捉位置信息。此外，我们引入神经热核来负责图数据处理，在GNN中利用隐藏层输出匹配来提高学生多层感知机的性能。据我们所知，这是首次在图上引入隐藏层蒸馏用于学生多层感知机，并结合图位置编码和多层感知机。我们通过多种设置测试了其性能和稳健性，并得出结论......

    arXiv:2403.01079v1 Announce Type: cross  Abstract: We study the challenging problem for inference tasks on large-scale graph datasets of Graph Neural Networks: huge time and memory consumption, and try to overcome it by reducing reliance on graph structure. Even though distilling graph knowledge to student MLP is an excellent idea, it faces two major problems of positional information loss and low generalization. To solve the problems, we propose a new three-stage multitask distillation framework. In detail, we use Positional Encoding to capture positional information. Also, we introduce Neural Heat Kernels responsible for graph data processing in GNN and utilize hidden layer outputs matching for better performance of student MLP's hidden layers. To the best of our knowledge, it is the first work to include hidden layer distillation for student MLP on graphs and to combine graph Positional Encoding with MLP. We test its performance and robustness with several settings and draw the conc
    
[^2]: ServeFlow：一种用于网络流量分析的快速-慢速模型架构

    ServeFlow: A Fast-Slow Model Architecture for Network Traffic Analysis

    [https://arxiv.org/abs/2402.03694](https://arxiv.org/abs/2402.03694)

    ServeFlow提出了一种快速-慢速模型架构，用于网络流量分析。通过精心选择收集数据包的数量和应用于不同流量的模型，ServeFlow实现了最小延迟、高服务率和高准确性之间的平衡。在测试中，ServeFlow能够在16ms内对76.3%的流量进行推理，这是中位数推理时间的40.5倍加速！

    

    随着互联网的整合和流量的加密，网络流量分析越来越多地使用复杂的机器学习模型。然而，在高带宽网络中，流量往往比模型的推理速率更快。网络流量的时间性质限制了在其他高流量机器学习应用中使用的简单扩展方法。因此，本文提出了ServeFlow，这是一个针对网络流量分析任务的机器学习模型提供解决方案，它通过精心选择收集数据包的数量和应用于不同流量的模型，来实现最小延迟、高服务率和高准确性之间的平衡。我们发现在相同的任务上，不同模型的推理时间可以相差2.7倍到136.3倍，而中位数数据包等待时间通常比推理时间高6到8个数量级！ServeFlow能够在16ms内对76.3%的流量进行推理，这是中位数推理时间的40.5倍加速！

    Network traffic analysis increasingly uses complex machine learning models as the internet consolidates and traffic gets more encrypted. However, over high-bandwidth networks, flows can easily arrive faster than model inference rates. The temporal nature of network flows limits simple scale-out approaches leveraged in other high-traffic machine learning applications. Accordingly, this paper presents ServeFlow, a solution for machine-learning model serving aimed at network traffic analysis tasks, which carefully selects the number of packets to collect and the models to apply for individual flows to achieve a balance between minimal latency, high service rate, and high accuracy. We identify that on the same task, inference time across models can differ by 2.7x-136.3x, while the median inter-packet waiting time is often 6-8 orders of magnitude higher than the inference time! ServeFlow is able to make inferences on 76.3% flows in under 16ms, which is a speed-up of 40.5x on the median end-
    
[^3]: 有限域情境演算理论中有关时间的可判定推理

    Decidable Reasoning About Time in Finite-Domain Situation Calculus Theories

    [https://arxiv.org/abs/2402.03164](https://arxiv.org/abs/2402.03164)

    本论文研究了有限域情境演算理论中关于时间的可判定推理。通过引入具有受限状态公理和比较操作符的钟表作为实值fluents，解决了检查可达情境是否满足给定公式的问题的不可判定性，并给出了一个可判定的过程来确定行动序列。

    

    时间的表达对于智能物理系统至关重要，并在情境演算中得到了广泛研究。最常用的方法是通过添加一个实值fluents $\mathit{time}(a)$ 来表示时间，这个fluents将一个时间点与每个动作以及每个情境相连。我们证明了在这种方法中，即使领域范围被限制在一个有限对象集上，检查是否存在一个可达情境满足给定公式的问题是不可判定的。我们提出了一种基于定时自动机理论的替代方法，通过引入钟表作为具有受限的继任状态公理和比较操作符的实值fluents。通过这种限制，我们可以证明有限域基本动作理论的可达性问题是可判定的。最后，我们将我们的结果应用于Golog程序实现，通过提供一种可判定的过程来确定一个行动序列

    Representing time is crucial for cyber-physical systems and has been studied extensively in the Situation Calculus. The most commonly used approach represents time by adding a real-valued fluent $\mathit{time}(a)$ that attaches a time point to each action and consequently to each situation. We show that in this approach, checking whether there is a reachable situation that satisfies a given formula is undecidable, even if the domain of discourse is restricted to a finite set of objects. We present an alternative approach based on well-established results from timed automata theory by introducing clocks as real-valued fluents with restricted successor state axioms and comparison operators. %that only allow comparisons against fixed rationals. With this restriction, we can show that the reachability problem for finite-domain basic action theories is decidable. Finally, we apply our results on Golog program realization by presenting a decidable procedure for determining an action sequence
    
[^4]: LLMRec: 在推荐任务上对大型语言模型进行基准测试

    LLMRec: Benchmarking Large Language Models on Recommendation Task. (arXiv:2308.12241v1 [cs.IR])

    [http://arxiv.org/abs/2308.12241](http://arxiv.org/abs/2308.12241)

    LLMRec是一个基于LLMs的推荐系统，用于对LLMs在推荐任务上进行基准测试。研究结果显示，LLMs在顺序推荐和直接推荐等准确性任务方面表现出中等熟练程度，并且在指令遵循能力方面具有与最先进方法相当的性能。

    

    近期，像ChatGPT这样的大型语言模型（LLMs）的快速发展通过增强对话模型的能力，显著提升了自然语言处理任务。然而，LLMs在推荐领域的应用尚未得到深入研究。为了填补这一空白，我们提出了LLMRec，这是一个基于LLMs的推荐系统，旨在对多种推荐任务进行LLMs的基准测试。具体而言，我们对几种热门的现成LLMs进行了基准测试，包括ChatGPT、LLaMA和ChatGLM，涵盖了评分预测、顺序推荐、直接推荐、解释生成和评论摘要等五个推荐任务。此外，我们研究了有监督微调的有效性，以提高LLMs的指令遵循能力。基准测试结果表明，LLMs在基于准确性的任务（如顺序推荐和直接推荐）上只表现出中等的熟练程度。然而，它们在性能上与最先进的方法控制能力相当。

    Recently, the fast development of Large Language Models (LLMs) such as ChatGPT has significantly advanced NLP tasks by enhancing the capabilities of conversational models. However, the application of LLMs in the recommendation domain has not been thoroughly investigated. To bridge this gap, we propose LLMRec, a LLM-based recommender system designed for benchmarking LLMs on various recommendation tasks. Specifically, we benchmark several popular off-the-shelf LLMs, such as ChatGPT, LLaMA, ChatGLM, on five recommendation tasks, including rating prediction, sequential recommendation, direct recommendation, explanation generation, and review summarization. Furthermore, we investigate the effectiveness of supervised finetuning to improve LLMs' instruction compliance ability. The benchmark results indicate that LLMs displayed only moderate proficiency in accuracy-based tasks such as sequential and direct recommendation. However, they demonstrated comparable performance to state-of-the-art me
    
[^5]: NeuSE: Neural SE(3)-等变嵌入用于一致的对象空间理解

    NeuSE: Neural SE(3)-Equivariant Embedding for Consistent Spatial Understanding with Objects. (arXiv:2303.07308v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2303.07308](http://arxiv.org/abs/2303.07308)

    NeuSE是一种用于对象的神经网络嵌入算法，支持实现与长期场景变化一致的对象空间理解。它可以编码完整的形状信息，并与物理世界中的对象同时变换，能够直接推断相对帧变换和相机姿态约束，并维持适应变化的轻量级对象地图。

    

    我们提出了NeuSE，这是一种用于对象的新颖的神经网络SE(3)-等变嵌入，并展示了它如何支持对象SLAM以实现与长期场景变化一致的空间理解。NeuSE是从部分对象观测中创建的一组潜在对象嵌入，在与物理世界中的对象同时SE(3)-等变变换的同时，它作为一个紧凑的点云替代完整的对象模型，编码了完整的形状信息。通过NeuSE，可以直接从推断的潜在代码中获得相对帧变换。我们提出的使用NeuSE进行对象形状和姿态特征化的SLAM范式可以独立运行或与典型的SLAM系统结合使用。它直接推断与普通SLAM姿态图优化兼容的SE(3)相机姿态约束，同时还维持了一个轻量级的以对象为中心的地图，能够适应现实世界的变化。我们的方法在包含改变的对象的合成和真实世界序列上进行了评估。

    We present NeuSE, a novel Neural SE(3)-Equivariant Embedding for objects, and illustrate how it supports object SLAM for consistent spatial understanding with long-term scene changes. NeuSE is a set of latent object embeddings created from partial object observations. It serves as a compact point cloud surrogate for complete object models, encoding full shape information while transforming SE(3)-equivariantly in tandem with the object in the physical world. With NeuSE, relative frame transforms can be directly derived from inferred latent codes. Our proposed SLAM paradigm, using NeuSE for object shape and pose characterization, can operate independently or in conjunction with typical SLAM systems. It directly infers SE(3) camera pose constraints that are compatible with general SLAM pose graph optimization, while also maintaining a lightweight object-centric map that adapts to real-world changes. Our approach is evaluated on synthetic and real-world sequences featuring changed objects 
    

