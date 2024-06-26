# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SoK: Facial Deepfake Detectors.](http://arxiv.org/abs/2401.04364) | 本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。 |
| [^2] | [A Probabilistic Fluctuation based Membership Inference Attack for Generative Models.](http://arxiv.org/abs/2308.12143) | 本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。 |
| [^3] | [Locally Differentially Private Distributed Online Learning with Guaranteed Optimality.](http://arxiv.org/abs/2306.14094) | 本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。 |

# 详细

[^1]: SoK：面部深度伪造检测器

    SoK: Facial Deepfake Detectors. (arXiv:2401.04364v1 [cs.CV])

    [http://arxiv.org/abs/2401.04364](http://arxiv.org/abs/2401.04364)

    本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。

    

    深度伪造技术迅速成为对社会构成深远和严重威胁的原因之一，主要由于其易于制作和传播。这种情况加速了深度伪造检测技术的发展。然而，许多现有的检测器在验证时 heavily 依赖实验室生成的数据集，这可能无法有效地让它们应对新颖、新兴和实际的深度伪造技术。本文对最新的深度伪造检测器进行广泛全面的回顾和分析，根据几个关键标准对它们进行评估。这些标准将这些检测器分为 4 个高级组别和 13 个细粒度子组别，都遵循一个统一的标准概念框架。这种分类和框架提供了对影响检测器功效的因素的深入和实用的见解。我们对 16 个主要的检测器在各种标准的攻击场景中的普适性进行评估，包括黑盒攻击场景。

    Deepfakes have rapidly emerged as a profound and serious threat to society, primarily due to their ease of creation and dissemination. This situation has triggered an accelerated development of deepfake detection technologies. However, many existing detectors rely heavily on lab-generated datasets for validation, which may not effectively prepare them for novel, emerging, and real-world deepfake techniques. In this paper, we conduct an extensive and comprehensive review and analysis of the latest state-of-the-art deepfake detectors, evaluating them against several critical criteria. These criteria facilitate the categorization of these detectors into 4 high-level groups and 13 fine-grained sub-groups, all aligned with a unified standard conceptual framework. This classification and framework offer deep and practical insights into the factors that affect detector efficacy. We assess the generalizability of 16 leading detectors across various standard attack scenarios, including black-bo
    
[^2]: 一种基于概率波动的生成模型成员推断攻击方法

    A Probabilistic Fluctuation based Membership Inference Attack for Generative Models. (arXiv:2308.12143v1 [cs.LG])

    [http://arxiv.org/abs/2308.12143](http://arxiv.org/abs/2308.12143)

    本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。

    

    成员推断攻击(MIA)通过查询模型来识别机器学习模型的训练集中是否存在某条记录。对经典分类模型的MIA已有很多研究，最近的工作开始探索如何将MIA应用到生成模型上。我们的研究表明，现有的面向生成模型的MIA主要依赖于目标模型的过拟合现象。然而，过拟合可以通过采用各种正则化技术来避免，而现有的MIA在实践中表现不佳。与过拟合不同，记忆对于深度学习模型实现最佳性能是至关重要的，使其成为一种更为普遍的现象。生成模型中的记忆导致生成记录的概率分布呈现出增长的趋势。因此，我们提出了一种基于概率波动的成员推断攻击方法(PFAMI)，它是一种黑盒MIA，通过检测概率波动来推断成员身份。

    Membership Inference Attack (MIA) identifies whether a record exists in a machine learning model's training set by querying the model. MIAs on the classic classification models have been well-studied, and recent works have started to explore how to transplant MIA onto generative models. Our investigation indicates that existing MIAs designed for generative models mainly depend on the overfitting in target models. However, overfitting can be avoided by employing various regularization techniques, whereas existing MIAs demonstrate poor performance in practice. Unlike overfitting, memorization is essential for deep learning models to attain optimal performance, making it a more prevalent phenomenon. Memorization in generative models leads to an increasing trend in the probability distribution of generating records around the member record. Therefore, we propose a Probabilistic Fluctuation Assessing Membership Inference Attack (PFAMI), a black-box MIA that infers memberships by detecting t
    
[^3]: 具有保证最优性的局部差分隐私分布式在线学习

    Locally Differentially Private Distributed Online Learning with Guaranteed Optimality. (arXiv:2306.14094v1 [cs.LG])

    [http://arxiv.org/abs/2306.14094](http://arxiv.org/abs/2306.14094)

    本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。

    

    分布式在线学习由于其处理大规模数据集和流数据的能力而受到越来越多的关注。为了解决隐私保护问题，已经提出了许多个人私密分布式在线学习算法，大多数基于差分隐私，差分隐私已成为隐私保护的“黄金标准”。然而，这些算法常常面临为了隐私保护而牺牲学习准确性的困境。本文利用在线学习的独特特征，提出了一种方法来解决这一困境，并确保分布式在线学习中的差分隐私和学习准确性。具体而言，该方法在确保预期瞬时遗憾程度逐渐减小的同时，还能保证有限的累积隐私预算，即使在无限时间范围内。为了应对完全分布式环境，我们采用本地差分隐私框架，避免了对全局数据的依赖。

    Distributed online learning is gaining increased traction due to its unique ability to process large-scale datasets and streaming data. To address the growing public awareness and concern on privacy protection, plenty of private distributed online learning algorithms have been proposed, mostly based on differential privacy which has emerged as the ``gold standard" for privacy protection. However, these algorithms often face the dilemma of trading learning accuracy for privacy. By exploiting the unique characteristics of online learning, this paper proposes an approach that tackles the dilemma and ensures both differential privacy and learning accuracy in distributed online learning. More specifically, while ensuring a diminishing expected instantaneous regret, the approach can simultaneously ensure a finite cumulative privacy budget, even on the infinite time horizon. To cater for the fully distributed setting, we adopt the local differential-privacy framework which avoids the reliance
    

