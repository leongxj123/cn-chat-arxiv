# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analyzing the Impact of Partial Sharing on the Resilience of Online Federated Learning Against Model Poisoning Attacks](https://arxiv.org/abs/2403.13108) | PSO-Fed算法的部分共享机制不仅可以降低通信负载，还能增强算法对模型投毒攻击的抵抗力，并且在面对拜占庭客户端的情况下依然能保持收敛。 |
| [^2] | [Attacking Graph Neural Networks with Bit Flips: Weisfeiler and Lehman Go Indifferent.](http://arxiv.org/abs/2311.01205) | 我们设计了Injectivity Bit Flip Attack来针对图神经网络，成功地降低了其对图结构的识别能力和表达能力，从而增加了其对位反转攻击的易受攻击性。 |
| [^3] | [S-GBDT: Frugal Differentially Private Gradient Boosting Decision Trees.](http://arxiv.org/abs/2309.12041) | S-GBDT是一种节俭的差分隐私梯度提升决策树学习器，利用了四种技术来改善效用和隐私权之间的平衡，包括对隐私泄露的更紧密计算和整合个体Rényi滤波器以学习未充分利用的数据点。 |
| [^4] | [Large Language Models for Code: Security Hardening and Adversarial Testing.](http://arxiv.org/abs/2302.05319) | 本研究针对大型语言模型在生成代码时缺乏安全意识，从安全加固和对抗测试的角度入手，提出了一项新的安全任务——受控代码生成，通过一种新型基于学习的方法SVEN，实现生成既安全又功能正确的代码，并对当前的LM进行对抗测试，强调了在LM的培训和评估中考虑安全因素的必要性。 |

# 详细

[^1]: 分析部分共享对在线联邦学习抵抗模型投毒攻击的影响

    Analyzing the Impact of Partial Sharing on the Resilience of Online Federated Learning Against Model Poisoning Attacks

    [https://arxiv.org/abs/2403.13108](https://arxiv.org/abs/2403.13108)

    PSO-Fed算法的部分共享机制不仅可以降低通信负载，还能增强算法对模型投毒攻击的抵抗力，并且在面对拜占庭客户端的情况下依然能保持收敛。

    

    我们审查了部分共享的在线联邦学习（PSO-Fed）算法对抵抗模型投毒攻击的韧性。 PSO-Fed通过使客户端在每个更新轮次仅与服务器交换部分模型估计来减少通信负载。模型估计的部分共享还增强了算法对模型投毒攻击的强度。为了更好地理解这一现象，我们分析了PSO-Fed算法在存在拜占庭客户端的情况下的性能，这些客户端可能会在与服务器共享之前通过添加噪声轻微篡改其本地模型。通过我们的分析，我们证明了PSO-Fed在均值和均方意义上都能保持收敛，即使在模型投毒攻击的压力下也是如此。我们进一步推导了PSO-Fed的理论均方误差（MSE），将其与步长、攻击概率、数字等各种参数联系起来。

    arXiv:2403.13108v1 Announce Type: new  Abstract: We scrutinize the resilience of the partial-sharing online federated learning (PSO-Fed) algorithm against model-poisoning attacks. PSO-Fed reduces the communication load by enabling clients to exchange only a fraction of their model estimates with the server at each update round. Partial sharing of model estimates also enhances the robustness of the algorithm against model-poisoning attacks. To gain better insights into this phenomenon, we analyze the performance of the PSO-Fed algorithm in the presence of Byzantine clients, malicious actors who may subtly tamper with their local models by adding noise before sharing them with the server. Through our analysis, we demonstrate that PSO-Fed maintains convergence in both mean and mean-square senses, even under the strain of model-poisoning attacks. We further derive the theoretical mean square error (MSE) of PSO-Fed, linking it to various parameters such as stepsize, attack probability, numb
    
[^2]: 使用位反转攻击图神经网络：Weisfeiler和Lehman变得冷漠了

    Attacking Graph Neural Networks with Bit Flips: Weisfeiler and Lehman Go Indifferent. (arXiv:2311.01205v1 [cs.LG])

    [http://arxiv.org/abs/2311.01205](http://arxiv.org/abs/2311.01205)

    我们设计了Injectivity Bit Flip Attack来针对图神经网络，成功地降低了其对图结构的识别能力和表达能力，从而增加了其对位反转攻击的易受攻击性。

    

    先前对图神经网络的攻击主要集中在图毒化和规避上，忽略了网络的权重和偏置。传统的基于权重的故障注入攻击，如用于卷积神经网络的位反转攻击，没有考虑到图神经网络的独特属性。我们提出了注入率位反转攻击（Injectivity Bit Flip Attack），这是第一个专门针对图神经网络设计的位反转攻击。我们的攻击针对量化消息传递神经网络中的可学习邻域聚合函数，降低了其区分图结构的能力，失去了Weisfeiler-Lehman测试的表达能力。我们的发现表明，利用特定于某些图神经网络架构的数学属性可能会显著增加其对位反转攻击的易受攻击性。注入率位反转攻击可以将各种图属性预测数据集上训练的最大表达性同构网络降级为随机输出。

    Prior attacks on graph neural networks have mostly focused on graph poisoning and evasion, neglecting the network's weights and biases. Traditional weight-based fault injection attacks, such as bit flip attacks used for convolutional neural networks, do not consider the unique properties of graph neural networks. We propose the Injectivity Bit Flip Attack, the first bit flip attack designed specifically for graph neural networks. Our attack targets the learnable neighborhood aggregation functions in quantized message passing neural networks, degrading their ability to distinguish graph structures and losing the expressivity of the Weisfeiler-Lehman test. Our findings suggest that exploiting mathematical properties specific to certain graph neural network architectures can significantly increase their vulnerability to bit flip attacks. Injectivity Bit Flip Attacks can degrade the maximal expressive Graph Isomorphism Networks trained on various graph property prediction datasets to rando
    
[^3]: S-GBDT: 节俭的差分隐私梯度提升决策树

    S-GBDT: Frugal Differentially Private Gradient Boosting Decision Trees. (arXiv:2309.12041v1 [cs.CR])

    [http://arxiv.org/abs/2309.12041](http://arxiv.org/abs/2309.12041)

    S-GBDT是一种节俭的差分隐私梯度提升决策树学习器，利用了四种技术来改善效用和隐私权之间的平衡，包括对隐私泄露的更紧密计算和整合个体Rényi滤波器以学习未充分利用的数据点。

    

    差分隐私学习梯度提升决策树(GBDT)在表格数据(如人口普查数据或医疗元数据)中具有很强的效用和隐私权之间的平衡潜力：经典的GBDT学习器可以从小规模数据集中提取非线性模式。可证明具有隐私性质的当前方法是差分隐私，该方法要求单个数据点的影响有限且可否认。我们引入了一种新的差分隐私GBDT学习器，并利用四种主要技术来改善效用和隐私权之间的平衡。(1)我们使用了一种改进的噪声缩放方法，更紧密地计算了与先前工作相比决策树叶子的隐私泄露，从而导致噪声的期望与数据点数量n的比例为$O(1/n)$，其中n为数据点数量。(2)我们将个体Rényi滤波器整合到我们的方法中，以从在迭代训练过程中未充分利用的数据点中学习，这可能是独立于兴趣的结果。

    Privacy-preserving learning of gradient boosting decision trees (GBDT) has the potential for strong utility-privacy tradeoffs for tabular data, such as census data or medical meta data: classical GBDT learners can extract non-linear patterns from small sized datasets. The state-of-the-art notion for provable privacy-properties is differential privacy, which requires that the impact of single data points is limited and deniable. We introduce a novel differentially private GBDT learner and utilize four main techniques to improve the utility-privacy tradeoff. (1) We use an improved noise scaling approach with tighter accounting of privacy leakage of a decision tree leaf compared to prior work, resulting in noise that in expectation scales with $O(1/n)$, for $n$ data points. (2) We integrate individual R\'enyi filters to our method to learn from data points that have been underutilized during an iterative training process, which -- potentially of independent interest -- results in a natura
    
[^4]: 用于编码的大型语言模型：安全加固和对抗测试

    Large Language Models for Code: Security Hardening and Adversarial Testing. (arXiv:2302.05319v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2302.05319](http://arxiv.org/abs/2302.05319)

    本研究针对大型语言模型在生成代码时缺乏安全意识，从安全加固和对抗测试的角度入手，提出了一项新的安全任务——受控代码生成，通过一种新型基于学习的方法SVEN，实现生成既安全又功能正确的代码，并对当前的LM进行对抗测试，强调了在LM的培训和评估中考虑安全因素的必要性。

    

    大型语言模型(LMs)越来越多地预先在大规模代码库上进行预处理，用于生成代码。然而，LM缺乏安全意识，并经常生成不安全的代码。本研究沿着两个重要方向研究了LM的安全性:(i)安全加固，旨在增强LM在生成安全代码方面的可靠性;(ii)对抗测试，旨在在对抗性立场评估LM的安全性。我们通过制定一项称为受控代码生成的新安全任务来同时解决这两个问题。该任务是参数化的，将一个二进制属性作为输入，以指导LM生成安全或不安全的代码，同时保留LM生成功能正确代码的能力。我们提出了一种称为SVEN的新型基于学习的方法来解决这个任务。SVEN利用属性特定的连续向量来引导程序生成达到给定的属性，而不修改LM的权重。我们的训练过程通过可微分的投影损失来优化这些连续向量，实现端到端的训练。此外，我们使用SVEN进行对抗测试，并表明当前的LM容易受到攻击，在测试时修改它们的输入而保留功能。我们的工作强调需要在LM的培训和评估中考虑安全因素。

    Large language models (LMs) are increasingly pretrained on massive codebases and used to generate code. However, LMs lack awareness of security and are found to frequently produce unsafe code. This work studies the security of LMs along two important axes: (i) security hardening, which aims to enhance LMs' reliability in generating secure code, and (ii) adversarial testing, which seeks to evaluate LMs' security at an adversarial standpoint. We address both of these by formulating a new security task called controlled code generation. The task is parametric and takes as input a binary property to guide the LM to generate secure or unsafe code, while preserving the LM's capability of generating functionally correct code. We propose a novel learning-based approach called SVEN to solve this task. SVEN leverages property-specific continuous vectors to guide program generation towards the given property, without modifying the LM's weights. Our training procedure optimizes these continuous ve
    

