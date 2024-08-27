# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710) | 介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。 |
| [^2] | [Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures](https://arxiv.org/abs/2403.14772) | 通过稀疏编码层设计新网络架构以提高对模型逆推攻击的鲁棒性。 |
| [^3] | [Continual Adversarial Defense](https://arxiv.org/abs/2312.09481) | 提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。 |
| [^4] | [Investigating Feature and Model Importance in Android Malware Detection: An Implemented Survey and Experimental Comparison of ML-Based Methods](https://arxiv.org/abs/2301.12778) | 本文重新实现和评估了18个代表性的过去研究，并在包含124,000个应用程序的平衡、相关和最新数据集上进行了新实验，发现仅通过静态分析提取的特征就能实现高达96.8%的恶意软件检测准确性。 |
| [^5] | [Multi-Trigger Backdoor Attacks: More Triggers, More Threats.](http://arxiv.org/abs/2401.15295) | 本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。 |
| [^6] | [Locally Differentially Private Distributed Online Learning with Guaranteed Optimality.](http://arxiv.org/abs/2306.14094) | 本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。 |
| [^7] | [MetaGAD: Learning to Meta Transfer for Few-shot Graph Anomaly Detection.](http://arxiv.org/abs/2305.10668) | 本文提出了一种名为MetaGAD的框架，用于学习从无标记节点到有标记节点之间的元转移知识，以进行少样本图异常检测。 |

# 详细

[^1]: 基于优化的对LLM评判系统的提示注入攻击

    Optimization-based Prompt Injection Attack to LLM-as-a-Judge

    [https://arxiv.org/abs/2403.17710](https://arxiv.org/abs/2403.17710)

    介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。

    

    LLM-as-a-Judge 是一种可以使用大型语言模型（LLMs）评估文本信息的新颖解决方案。根据现有研究，LLMs在提供传统人类评估的引人注目替代方面表现出色。然而，这些系统针对提示注入攻击的鲁棒性仍然是一个未解决的问题。在这项工作中，我们引入了JudgeDeceiver，一种针对LLM-as-a-Judge量身定制的基于优化的提示注入攻击。我们的方法制定了一个精确的优化目标，用于攻击LLM-as-a-Judge的决策过程，并利用优化算法高效地自动化生成对抗序列，实现对模型评估的有针对性和有效的操作。与手工制作的提示注入攻击相比，我们的方法表现出卓越的功效，给基于LLM的判断系统当前的安全范式带来了重大挑战。

    arXiv:2403.17710v1 Announce Type: cross  Abstract: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. T
    
[^2]: 通过稀疏编码架构提高模型逆推攻击的鲁棒性

    Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures

    [https://arxiv.org/abs/2403.14772](https://arxiv.org/abs/2403.14772)

    通过稀疏编码层设计新网络架构以提高对模型逆推攻击的鲁棒性。

    

    最近的模型逆推攻击算法允许对手通过反复查询神经网络并检查其输出来重建网络的私有训练数据。 在这项工作中，我们开发了一种新颖的网络架构，利用稀疏编码层来获得对这类攻击的卓越鲁棒性。 三十年来，计算机科学研究已经研究了稀疏编码在图像去噪，目标识别和对抗性误分设置中的作用，但据我们所知，其与最先进的隐私漏洞之间的联系尚未被研究。然而，稀疏编码架构提供了一种有利的手段来抵御模型逆推攻击，因为它们允许我们控制编码在网络的中间表示中的无关私人信息的数量，而这种方式可以在训练过程中高效计算，并且众所周知只有较小的影响。

    arXiv:2403.14772v1 Announce Type: cross  Abstract: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private training data just by repeatedly querying the network and inspecting its outputs. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. However, sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded in a network's intermediate representations in a manner that can be computed efficiently during training and that is known to have little effect
    
[^3]: 持续不断的对抗性防御

    Continual Adversarial Defense

    [https://arxiv.org/abs/2312.09481](https://arxiv.org/abs/2312.09481)

    提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。

    

    针对每月针对视觉分类器的对抗性攻击快速演变的特性，人们提出了许多防御方法，旨在尽可能通用化以抵御尽可能多的已知攻击。然而，设计一个能够对抗所有类型攻击的防御方法并不现实，因为防御系统运行的环境是动态的，包含随着时间出现的各种独特攻击。防御系统必须收集在线少样本对抗反馈以迅速增强自身，充分利用内存。因此，我们提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架，其中各种攻击逐个阶段出现。在实践中，CAD基于四项原则进行建模：(1) 持续适应新攻击而无灾难性遗忘，(2) 少样本适应，(3) 内存高效适应，以及(4) 高准确性

    arXiv:2312.09481v2 Announce Type: replace-cross  Abstract: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accur
    
[^4]: 探究安卓恶意软件检测中的特征和模型重要性：一项实施调查和机器学习方法实验比较

    Investigating Feature and Model Importance in Android Malware Detection: An Implemented Survey and Experimental Comparison of ML-Based Methods

    [https://arxiv.org/abs/2301.12778](https://arxiv.org/abs/2301.12778)

    本文重新实现和评估了18个代表性的过去研究，并在包含124,000个应用程序的平衡、相关和最新数据集上进行了新实验，发现仅通过静态分析提取的特征就能实现高达96.8%的恶意软件检测准确性。

    

    Android的普及意味着它成为恶意软件的常见目标。多年来，各种研究发现机器学习模型能够有效区分恶意软件和良性应用程序。然而，随着操作系统的演进，恶意软件也在不断发展，对先前研究的发现提出了质疑，其中许多报告称使用小型、过时且经常不平衡的数据集能够获得非常高的准确性。在本文中，我们重新实现了18项具代表性的过去研究并使用包括124,000个应用程序的平衡、相关且最新的数据集对它们进行重新评估。我们还进行了新的实验，以填补现有知识中的空白，并利用研究结果确定在当代环境中用于安卓恶意软件检测的最有效特征和模型。我们表明，仅通过静态分析提取的特征即可实现高达96.8%的检测准确性。

    arXiv:2301.12778v2 Announce Type: replace  Abstract: The popularity of Android means it is a common target for malware. Over the years, various studies have found that machine learning models can effectively discriminate malware from benign applications. However, as the operating system evolves, so does malware, bringing into question the findings of these previous studies, many of which report very high accuracies using small, outdated, and often imbalanced datasets. In this paper, we reimplement 18 representative past works and reevaluate them using a balanced, relevant, and up-to-date dataset comprising 124,000 applications. We also carry out new experiments designed to fill holes in existing knowledge, and use our findings to identify the most effective features and models to use for Android malware detection within a contemporary environment. We show that high detection accuracies (up to 96.8%) can be achieved using features extracted through static analysis alone, yielding a mode
    
[^5]: 多触发后门攻击：更多触发器，更多威胁

    Multi-Trigger Backdoor Attacks: More Triggers, More Threats. (arXiv:2401.15295v1 [cs.LG])

    [http://arxiv.org/abs/2401.15295](http://arxiv.org/abs/2401.15295)

    本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。

    

    后门攻击已经成为深度神经网络（DNNs）的（预）训练和部署的主要威胁。尽管后门攻击在一些研究中已经得到了广泛的探讨，但其中大部分都集中在使用单个类型的触发器来污染数据集的单触发攻击上。可以说，在现实世界中，后门攻击可能更加复杂，例如，同一数据集可能存在多个对手，如果该数据集具有较高的价值。在这项工作中，我们研究了在多触发攻击设置下后门攻击的实际威胁，多个对手利用不同类型的触发器来污染同一数据集。通过提出和研究并行、顺序和混合攻击这三种类型的多触发攻击，我们提供了关于不同触发器对同一数据集的共存、覆写和交叉激活效果的重要认识。此外，我们还展示了单触发攻击往往容易引起覆写问题。

    Backdoor attacks have emerged as a primary threat to (pre-)training and deployment of deep neural networks (DNNs). While backdoor attacks have been extensively studied in a body of works, most of them were focused on single-trigger attacks that poison a dataset using a single type of trigger. Arguably, real-world backdoor attacks can be much more complex, e.g., the existence of multiple adversaries for the same dataset if it is of high value. In this work, we investigate the practical threat of backdoor attacks under the setting of \textbf{multi-trigger attacks} where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks, including parallel, sequential, and hybrid attacks, we provide a set of important understandings of the coexisting, overwriting, and cross-activating effects between different triggers on the same dataset. Moreover, we show that single-trigger attacks tend to cause over
    
[^6]: 具有保证最优性的局部差分隐私分布式在线学习

    Locally Differentially Private Distributed Online Learning with Guaranteed Optimality. (arXiv:2306.14094v1 [cs.LG])

    [http://arxiv.org/abs/2306.14094](http://arxiv.org/abs/2306.14094)

    本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。

    

    分布式在线学习由于其处理大规模数据集和流数据的能力而受到越来越多的关注。为了解决隐私保护问题，已经提出了许多个人私密分布式在线学习算法，大多数基于差分隐私，差分隐私已成为隐私保护的“黄金标准”。然而，这些算法常常面临为了隐私保护而牺牲学习准确性的困境。本文利用在线学习的独特特征，提出了一种方法来解决这一困境，并确保分布式在线学习中的差分隐私和学习准确性。具体而言，该方法在确保预期瞬时遗憾程度逐渐减小的同时，还能保证有限的累积隐私预算，即使在无限时间范围内。为了应对完全分布式环境，我们采用本地差分隐私框架，避免了对全局数据的依赖。

    Distributed online learning is gaining increased traction due to its unique ability to process large-scale datasets and streaming data. To address the growing public awareness and concern on privacy protection, plenty of private distributed online learning algorithms have been proposed, mostly based on differential privacy which has emerged as the ``gold standard" for privacy protection. However, these algorithms often face the dilemma of trading learning accuracy for privacy. By exploiting the unique characteristics of online learning, this paper proposes an approach that tackles the dilemma and ensures both differential privacy and learning accuracy in distributed online learning. More specifically, while ensuring a diminishing expected instantaneous regret, the approach can simultaneously ensure a finite cumulative privacy budget, even on the infinite time horizon. To cater for the fully distributed setting, we adopt the local differential-privacy framework which avoids the reliance
    
[^7]: MetaGAD：学习元转移进行少样本图异常检测

    MetaGAD: Learning to Meta Transfer for Few-shot Graph Anomaly Detection. (arXiv:2305.10668v1 [cs.LG])

    [http://arxiv.org/abs/2305.10668](http://arxiv.org/abs/2305.10668)

    本文提出了一种名为MetaGAD的框架，用于学习从无标记节点到有标记节点之间的元转移知识，以进行少样本图异常检测。

    

    图异常检测长期以来一直是各个领域信息安全问题中的重要问题，如金融欺诈、社会垃圾邮件、网络入侵等。目前大多数现有方法都是以无监督方式执行的，因为标记的异常在大规模情况下往往太昂贵。然而，由于缺乏有关异常的先前知识，可能会将被识别的异常视为数据噪声或不感兴趣的数据实例。在现实场景中，通常可获取有限的标记异常，这些标记异常具有推进图异常检测的巨大潜力。然而，探索少量标记异常和大量无标记节点来检测异常的工作相当有限。因此，本文研究了少样本图异常检测的新问题。我们提出了一种新的框架MetaGAD，学习元转移知识来进行图异常检测。实

    Graph anomaly detection has long been an important problem in various domains pertaining to information security such as financial fraud, social spam, network intrusion, etc. The majority of existing methods are performed in an unsupervised manner, as labeled anomalies in a large scale are often too expensive to acquire. However, the identified anomalies may turn out to be data noises or uninteresting data instances due to the lack of prior knowledge on the anomalies. In realistic scenarios, it is often feasible to obtain limited labeled anomalies, which have great potential to advance graph anomaly detection. However, the work exploring limited labeled anomalies and a large amount of unlabeled nodes in graphs to detect anomalies is rather limited. Therefore, in this paper, we study a novel problem of few-shot graph anomaly detection. We propose a new framework MetaGAD to learn to meta-transfer the knowledge between unlabeled and labeled nodes for graph anomaly detection. Experimental 
    

