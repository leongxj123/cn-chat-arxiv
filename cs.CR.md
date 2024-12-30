# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Mathematical Framework for the Problem of Security for Cognition in Neurotechnology](https://arxiv.org/abs/2403.07945) | 本文提出了一个数学框架，名为认知安全，用于描述和分析神经技术对个体认知隐私和自治可能产生的影响，解决了相关问题描述和分析的障碍。 |
| [^2] | [A Probabilistic Fluctuation based Membership Inference Attack for Generative Models.](http://arxiv.org/abs/2308.12143) | 本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。 |

# 详细

[^1]: 一个解决神经技术认知安全问题的数学框架

    A Mathematical Framework for the Problem of Security for Cognition in Neurotechnology

    [https://arxiv.org/abs/2403.07945](https://arxiv.org/abs/2403.07945)

    本文提出了一个数学框架，名为认知安全，用于描述和分析神经技术对个体认知隐私和自治可能产生的影响，解决了相关问题描述和分析的障碍。

    

    近年来神经技术的快速发展在神经技术和安全之间创造了一个新兴的关键交叉点。植入式设备、非侵入式监测和非侵入式治疗都带来了违反个体认知隐私和自治的前景。越来越多的科学家和医生呼吁解决这一问题 -- 我们称之为认知安全 -- 但应用工作受到限制。阻碍科学和工程努力解决认知安全问题的一个主要障碍是缺乏清晰描述和分析相关问题的手段。在本文中，我们开发了认知安全，这是一个数学框架，通过借鉴多个领域的方法和结果，实现这种描述和分析。我们展示了一些对认知安全有重要影响的统计特性，然后提出描述...

    arXiv:2403.07945v1 Announce Type: cross  Abstract: The rapid advancement in neurotechnology in recent years has created an emerging critical intersection between neurotechnology and security. Implantable devices, non-invasive monitoring, and non-invasive therapies all carry with them the prospect of violating the privacy and autonomy of individuals' cognition. A growing number of scientists and physicians have made calls to address this issue -- which we term Cognitive Security -- but applied efforts have been limited. A major barrier hampering scientific and engineering efforts to address Cognitive Security is the lack of a clear means of describing and analyzing relevant problems. In this paper we develop Cognitive Security, a mathematical framework which enables such description and analysis by drawing on methods and results from multiple fields. We demonstrate certain statistical properties which have significant implications for Cognitive Security, and then present descriptions of
    
[^2]: 一种基于概率波动的生成模型成员推断攻击方法

    A Probabilistic Fluctuation based Membership Inference Attack for Generative Models. (arXiv:2308.12143v1 [cs.LG])

    [http://arxiv.org/abs/2308.12143](http://arxiv.org/abs/2308.12143)

    本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。

    

    成员推断攻击(MIA)通过查询模型来识别机器学习模型的训练集中是否存在某条记录。对经典分类模型的MIA已有很多研究，最近的工作开始探索如何将MIA应用到生成模型上。我们的研究表明，现有的面向生成模型的MIA主要依赖于目标模型的过拟合现象。然而，过拟合可以通过采用各种正则化技术来避免，而现有的MIA在实践中表现不佳。与过拟合不同，记忆对于深度学习模型实现最佳性能是至关重要的，使其成为一种更为普遍的现象。生成模型中的记忆导致生成记录的概率分布呈现出增长的趋势。因此，我们提出了一种基于概率波动的成员推断攻击方法(PFAMI)，它是一种黑盒MIA，通过检测概率波动来推断成员身份。

    Membership Inference Attack (MIA) identifies whether a record exists in a machine learning model's training set by querying the model. MIAs on the classic classification models have been well-studied, and recent works have started to explore how to transplant MIA onto generative models. Our investigation indicates that existing MIAs designed for generative models mainly depend on the overfitting in target models. However, overfitting can be avoided by employing various regularization techniques, whereas existing MIAs demonstrate poor performance in practice. Unlike overfitting, memorization is essential for deep learning models to attain optimal performance, making it a more prevalent phenomenon. Memorization in generative models leads to an increasing trend in the probability distribution of generating records around the member record. Therefore, we propose a Probabilistic Fluctuation Assessing Membership Inference Attack (PFAMI), a black-box MIA that infers memberships by detecting t
    

