# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the resilience of Collaborative Learning-based Recommender Systems Against Community Detection Attack](https://arxiv.org/abs/2306.08929) | 本文研究了协作学习推荐系统针对一种新型隐私攻击——社区检测攻击（CDA）的韧性。 |
| [^2] | [Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings.](http://arxiv.org/abs/2401.09727) | 本研究比较了大型语言模型在大规模组织环境中实现横向网络钓鱼的情况，并发现现有的反钓鱼基础设施无法防止语言模型生成的钓鱼攻击。 |
| [^3] | [Privacy-Preserving CNN Training with Transfer Learning.](http://arxiv.org/abs/2304.03807) | 本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。 |

# 详细

[^1]: 关于协作学习推荐系统针对社区检测攻击的韧性研究

    On the resilience of Collaborative Learning-based Recommender Systems Against Community Detection Attack

    [https://arxiv.org/abs/2306.08929](https://arxiv.org/abs/2306.08929)

    本文研究了协作学习推荐系统针对一种新型隐私攻击——社区检测攻击（CDA）的韧性。

    

    协作学习推荐系统源于协作学习技术（如联邦学习和八卦学习）的成功。在这些系统中，用户参与推荐系统的训练同时在其设备上保留已消费项目的历史记录。虽然这些解决方案乍一看似乎有利于保护参与者的隐私，但最近的研究表明，协作学习可能容易受到各种隐私攻击的威胁。本文研究了协作学习推荐系统针对一种称为社区检测攻击（CDA）的新型隐私攻击的韧性。这种攻击使得对手能够基于一个选择的项目集（如识别对特定兴趣点感兴趣的用户）来识别社区成员。通过在三个真实推荐数据集上进行实验，使用两种最先进的推荐

    arXiv:2306.08929v2 Announce Type: replace-cross  Abstract: Collaborative-learning-based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while maintaining their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at first glance, recent studies have revealed that collaborative learning can be vulnerable to various privacy attacks. In this paper, we study the resilience of collaborative learning-based recommender systems against a novel privacy attack called Community Detection Attack (CDA). This attack enables an adversary to identify community members based on a chosen set of items (eg., identifying users interested in specific points-of-interest). Through experiments on three real recommendation datasets using two state-of-the-art recomme
    
[^2]: 大型语言模型横向网络钓鱼：大规模组织环境中的比较研究

    Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings. (arXiv:2401.09727v1 [cs.CR])

    [http://arxiv.org/abs/2401.09727](http://arxiv.org/abs/2401.09727)

    本研究比较了大型语言模型在大规模组织环境中实现横向网络钓鱼的情况，并发现现有的反钓鱼基础设施无法防止语言模型生成的钓鱼攻击。

    

    钓鱼电子邮件的严重威胁被LLMs生成高度定向、个性化和自动化的鱼叉式网络钓鱼攻击的潜力进一步恶化。关于LLM促成的钓鱼存在两个关键问题需要进一步调查：1）现有的横向网络钓鱼研究缺乏针对整个组织进行大规模攻击的LLM整合的具体审查；2）尽管反钓鱼基础设施经过广泛开发，但仍无法防止LLM生成的攻击，可能影响员工和IT安全事件管理。然而，进行这样的调查研究需要在现实世界环境中进行，该环境在正常业务运作期间工作，并反映出大型组织基础设施的复杂性。此设置还必须提供所需的灵活性，以促进各种实验条件的实施，特别是钓鱼电子邮件的制作和组织范围的攻击。

    The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted
    
[^3]: 使用迁移学习实现隐私保护的CNN训练

    Privacy-Preserving CNN Training with Transfer Learning. (arXiv:2304.03807v1 [cs.CR])

    [http://arxiv.org/abs/2304.03807](http://arxiv.org/abs/2304.03807)

    本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。

    

    隐私保护的神经网络推理已经得到很好的研究，同时保持同态CNN训练仍然是一项挑战性的任务。在本文中，我们提出了一种实用的解决方案来实现基于同态加密技术的隐私保护CNN训练。据我们所知，这是第一次成功突破这个难题，以前没有任何工作达到这个目标。采用了几种技术：（1）通过迁移学习，可以将隐私保护的CNN训练简化为同态神经网络训练，甚至是多类逻辑回归（MLR）训练；（2）通过更快的梯度变体$\texttt{Quadratic Gradient}$，应用于MLR的增强梯度方法，在收敛速度方面具有最先进的性能；（3）我们采用数学中的变换思想，将加密域中的近似Softmax函数转换成已经研究过的逼近方法，从而得到更好的结果。

    Privacy-preserving nerual network inference has been well studied while homomorphic CNN training still remains an open challenging task. In this paper, we present a practical solution to implement privacy-preserving CNN training based on mere Homomorphic Encryption (HE) technique. To our best knowledge, this is the first attempt successfully to crack this nut and no work ever before has achieved this goal. Several techniques combine to make it done: (1) with transfer learning, privacy-preserving CNN training can be reduced to homomorphic neural network training, or even multiclass logistic regression (MLR) training; (2) via a faster gradient variant called $\texttt{Quadratic Gradient}$, an enhanced gradient method for MLR with a state-of-the-art performance in converge speed is applied in this work to achieve high performance; (3) we employ the thought of transformation in mathematics to transform approximating Softmax function in encryption domain to the well-studied approximation of 
    

