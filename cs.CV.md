# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [WebLINX: Real-World Website Navigation with Multi-Turn Dialogue](https://arxiv.org/abs/2402.05930) | 本论文提出了对话式网站导航的问题，并设计了一个 WEBLINX 基准测试，用于训练和评估代理。为了解决大量信息的处理瓶颈，文中提出了一个受检索启发的模型。实验结果表明，该模型能够在多种场景下复制人类行为的能力。 |
| [^3] | [Re-parameterized Low-rank Prompt: Generalize a Vision-Language Model within 0.5K Parameters.](http://arxiv.org/abs/2312.10813) | 该论文提出了一种新型的提示方法，重新参数化低秩提示（RLP），用于在大型预训练视觉语言模型的适应过程中实现高效和有效的知识转移。该方法能够显著减少可调参数和存储开销。 |
| [^4] | [Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review.](http://arxiv.org/abs/2308.05731) | 这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。 |
| [^5] | [The Principle of Uncertain Maximum Entropy.](http://arxiv.org/abs/2305.09868) | 介绍了不确定最大熵原理，该原理可以处理模型元素不可观测的情况，并优于特定条件下的最大熵方法。同时将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，性能得到了提高。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: WebLINX: 多轮对话下的真实世界网站导航

    WebLINX: Real-World Website Navigation with Multi-Turn Dialogue

    [https://arxiv.org/abs/2402.05930](https://arxiv.org/abs/2402.05930)

    本论文提出了对话式网站导航的问题，并设计了一个 WEBLINX 基准测试，用于训练和评估代理。为了解决大量信息的处理瓶颈，文中提出了一个受检索启发的模型。实验结果表明，该模型能够在多种场景下复制人类行为的能力。

    

    我们提出了对话式网站导航的问题，其中数字代理控制着一个网页浏览器，并按照用户的指令以多轮对话的方式解决真实世界任务。为了支持这个问题，我们引入了 WEBLINX - 一个100K交互的大规模基准测试，在2300个专家演示中进行了对话式网站导航的测试。我们的基准涵盖了150多个真实世界网站上的广泛模式，可以用于在不同场景下训练和评估代理。由于存在大量信息，大型语言模型 (LLMs) 无法实时处理整个网页。为了解决这个瓶颈，我们设计了一个受检索启发的模型，通过排名相关元素来高效地修剪 HTML 页面。我们使用选定的元素，以及屏幕截图和操作历史记录，评估各种模型在导航网页时复制人类行为的能力。我们的实验从小型纯文本模型到专有的多模态 LLMs 进行了测试。

    We propose the problem of conversational web navigation, where a digital agent controls a web browser and follows user instructions to solve real-world tasks in a multi-turn dialogue fashion. To support this problem, we introduce WEBLINX - a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. Our benchmark covers a broad range of patterns on over 150 real-world websites and can be used to train and evaluate agents in diverse scenarios. Due to the magnitude of information present, Large Language Models (LLMs) cannot process entire web pages in real-time. To solve this bottleneck, we design a retrieval-inspired model that efficiently prunes HTML pages by ranking relevant elements. We use the selected elements, along with screenshots and action history, to assess a variety of models for their ability to replicate human behavior when navigating the web. Our experiments span from small text-only to proprietary multimodal LLMs. We fi
    
[^3]: 重新参数化低秩提示：在0.5K参数内推广视觉语言模型

    Re-parameterized Low-rank Prompt: Generalize a Vision-Language Model within 0.5K Parameters. (arXiv:2312.10813v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.10813](http://arxiv.org/abs/2312.10813)

    该论文提出了一种新型的提示方法，重新参数化低秩提示（RLP），用于在大型预训练视觉语言模型的适应过程中实现高效和有效的知识转移。该方法能够显著减少可调参数和存储开销。

    

    随着大型预训练视觉语言模型的发展，如何有效地将这些基础模型的知识转移到下游任务中成为一个热门话题，尤其是在数据不足的情况下。最近，提示调优已成为一种流行的解决方案。在调整视觉语言模型时，研究人员冻结骨干部分的参数，只设计和调整提示。一方面，提示调优的精心设计展现出强大的性能。另一方面，复杂的结构和更新规则大大增加了计算和存储成本。受到观察到的视觉语言模型中泛化能力的演变模式与适应过程中提示矩阵秩变化趋势的调和一致性的启发，我们设计了一种新型提示，重新参数化低秩提示（RLP），用于高效和有效的适应。我们的方法能大大减少可调参数和存储开销。

    With the development of large pre-trained vision-language models, how to effectively transfer the knowledge of such foundational models to downstream tasks becomes a hot topic, especially in a data-deficient scenario. Recently, prompt tuning has become a popular solution. When adapting the vision-language models, researchers freeze the parameters in the backbone and only design and tune the prompts. On the one hand, the delicate design of prompt tuning exhibits strong performance. On the other hand, complicated structures and update rules largely increase the computation and storage cost. Motivated by the observation that the evolution pattern of the generalization capability in visual-language models aligns harmoniously with the trend of rank variations in the prompt matrix during adaptation, we design a new type of prompt, Re-parameterized Low-rank Prompt (RLP), for both efficient and effective adaptation. Our method could largely reduce the number of tunable parameters and storage s
    
[^4]: 重新思考基于深度学习的自动驾驶系统中的预测和规划的整合：一项综述

    Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review. (arXiv:2308.05731v1 [cs.RO])

    [http://arxiv.org/abs/2308.05731](http://arxiv.org/abs/2308.05731)

    这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。

    

    自动驾驶有可能彻底改变个人、公共和货运交通的方式。除了感知环境的巨大挑战外，即准确地使用可用的传感器数据感知环境，自动驾驶还包括规划一个安全、舒适和高效的运动轨迹。为了促进安全和进步，许多工作依赖于模块化的交通未来运动的预测。模块化的自动驾驶系统通常将预测和规划作为顺序的独立任务处理。虽然这考虑了周围交通对自车的影响，但它未能预测交通参与者对自车行为的反应。最近的研究表明，将预测和规划整合为相互依赖的联合步骤是实现安全、高效和舒适驾驶所必需的。虽然有各种模型实现了这种集成系统，但对不同原理的全面概述和理论理解仍然缺乏。

    Automated driving has the potential to revolutionize personal, public, and freight mobility. Besides the enormous challenge of perception, i.e. accurately perceiving the environment using available sensor data, automated driving comprises planning a safe, comfortable, and efficient motion trajectory. To promote safety and progress, many works rely on modules that predict the future motion of surrounding traffic. Modular automated driving systems commonly handle prediction and planning as sequential separate tasks. While this accounts for the influence of surrounding traffic on the ego-vehicle, it fails to anticipate the reactions of traffic participants to the ego-vehicle's behavior. Recent works suggest that integrating prediction and planning in an interdependent joint step is necessary to achieve safe, efficient, and comfortable driving. While various models implement such integrated systems, a comprehensive overview and theoretical understanding of different principles are lacking.
    
[^5]: 不确定最大熵原理

    The Principle of Uncertain Maximum Entropy. (arXiv:2305.09868v1 [cs.IT])

    [http://arxiv.org/abs/2305.09868](http://arxiv.org/abs/2305.09868)

    介绍了不确定最大熵原理，该原理可以处理模型元素不可观测的情况，并优于特定条件下的最大熵方法。同时将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，性能得到了提高。

    

    最大熵原理在信息理论中的引入，为统计力学，机器学习和生态学等各个领域的发展做出了贡献。其得到的解决方案作为催化剂，促进研究人员将他们的经验观察映射到获取无偏模型，同时加深了对复杂系统和现象的理解。然而，在模型元素不直接可观测的情况下，例如存在噪声或眼部遮挡的情况下，标准最大熵方法可能会失败，因为它们无法匹配特征约束。在这里，我们展示了不确定最大熵原理作为一种方法，尽管存在任意噪声观察，它同时将所有可用信息编码，而且优于一些特定条件下的最大熵方法的准确度。此外，我们将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，从而在与最大似然算法相比时建立了改进的性能。

    The principle of maximum entropy, as introduced by Jaynes in information theory, has contributed to advancements in various domains such as Statistical Mechanics, Machine Learning, and Ecology. Its resultant solutions have served as a catalyst, facilitating researchers in mapping their empirical observations to the acquisition of unbiased models, whilst deepening the understanding of complex systems and phenomena. However, when we consider situations in which the model elements are not directly observable, such as when noise or ocular occlusion is present, possibilities arise for which standard maximum entropy approaches may fail, as they are unable to match feature constraints. Here we show the Principle of Uncertain Maximum Entropy as a method that both encodes all available information in spite of arbitrarily noisy observations while surpassing the accuracy of some ad-hoc methods. Additionally, we utilize the output of a black-box machine learning model as input into an uncertain ma
    

