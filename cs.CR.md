# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Secret Collusion Among Generative AI Agents](https://arxiv.org/abs/2402.07510) | 本文汇集了人工智能和安全领域的相关概念，系统地形式化了生成式AI代理系统中的秘密勾结问题，并提出了缓解措施。通过测试各种形式的秘密勾结所需的能力，我们发现当前模型的隐写能力有限，但 GPT-4 展示了能力的飞跃。 |
| [^2] | [Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach](https://arxiv.org/abs/2402.02672) | 本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。 |
| [^3] | [Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition.](http://arxiv.org/abs/2401.10337) | 该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。 |

# 详细

[^1]: 生成式AI代理之间的秘密勾结

    Secret Collusion Among Generative AI Agents

    [https://arxiv.org/abs/2402.07510](https://arxiv.org/abs/2402.07510)

    本文汇集了人工智能和安全领域的相关概念，系统地形式化了生成式AI代理系统中的秘密勾结问题，并提出了缓解措施。通过测试各种形式的秘密勾结所需的能力，我们发现当前模型的隐写能力有限，但 GPT-4 展示了能力的飞跃。

    

    最近大型语言模型在能力上的增强为通信的生成式AI代理团队解决联合任务的应用打开了可能性。这引发了关于未经授权分享信息或其他不必要的代理协调形式的隐私和安全挑战。现代隐写术技术可能使这种动态难以检测。本文通过汲取人工智能和安全领域相关概念，全面系统地形式化了生成式AI代理系统中的秘密勾结问题。我们研究了使用隐写术的动机，并提出了各种缓解措施。我们的研究结果是一个模型评估框架，系统地测试了各种形式的秘密勾结所需的能力。我们在各种当代大型语言模型上提供了广泛的实证结果。虽然当前模型的隐写能力仍然有限，但 GPT-4 显示出能力的飞跃，这表明有必要进行进一步的研究。

    Recent capability increases in large language models (LLMs) open up applications in which teams of communicating generative AI agents solve joint tasks. This poses privacy and security challenges concerning the unauthorised sharing of information, or other unwanted forms of agent coordination. Modern steganographic techniques could render such dynamics hard to detect. In this paper, we comprehensively formalise the problem of secret collusion in systems of generative AI agents by drawing on relevant concepts from both the AI and security literature. We study incentives for the use of steganography, and propose a variety of mitigation measures. Our investigations result in a model evaluation framework that systematically tests capabilities required for various forms of secret collusion. We provide extensive empirical results across a range of contemporary LLMs. While the steganographic capabilities of current models remain limited, GPT-4 displays a capability jump suggesting the need fo
    
[^2]: 对分布式数据的条件平均治疗效果估计：一种保护隐私的方法

    Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach

    [https://arxiv.org/abs/2402.02672](https://arxiv.org/abs/2402.02672)

    本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。

    

    在医学和社会科学等各个领域中，对条件平均治疗效果（CATEs）的估计是一个重要的课题。如果分布在多个参与方之间的数据可以集中，可以对CATEs进行高精度的估计。然而，如果这些数据包含隐私信息，则很难进行数据聚合。为了解决这个问题，我们提出了数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计CATE模型，并通过数值实验对该方法进行了评估。我们的贡献总结如下三点。首先，我们的方法能够在分布式数据上进行非迭代通信的半参数CATE模型的估计和测试。半参数或非参数的CATE模型能够比参数模型更稳健地进行估计和测试，对于模型偏差的鲁棒性更强。然而，据我们所知，目前还没有提出有效的通信方法来估计和测试这些模型。

    Estimation of conditional average treatment effects (CATEs) is an important topic in various fields such as medical and social sciences. CATEs can be estimated with high accuracy if distributed data across multiple parties can be centralized. However, it is difficult to aggregate such data if they contain privacy information. To address this issue, we proposed data collaboration double machine learning (DC-DML), a method that can estimate CATE models with privacy preservation of distributed data, and evaluated the method through numerical experiments. Our contributions are summarized in the following three points. First, our method enables estimation and testing of semi-parametric CATE models without iterative communication on distributed data. Semi-parametric or non-parametric CATE models enable estimation and testing that is more robust to model mis-specification than parametric models. However, to our knowledge, no communication-efficient method has been proposed for estimating and 
    
[^3]: 基于噪声对比估计的低资源安全攻击模式识别匹配框架

    Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition. (arXiv:2401.10337v1 [cs.LG])

    [http://arxiv.org/abs/2401.10337](http://arxiv.org/abs/2401.10337)

    该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。

    

    战术、技术和程序（TTPs）是网络安全领域中复杂的攻击模式，在文本知识库中有详细的描述。在网络安全写作中识别TTPs，通常称为TTP映射，是一个重要而具有挑战性的任务。传统的学习方法通常以经典的多类或多标签分类设置为目标。由于存在大量的类别（即TTPs），标签分布的不均衡和标签空间的复杂层次结构，这种设置限制了模型的学习能力。我们采用了一种不同的学习范式来解决这个问题，其中将文本与TTP标签之间的直接语义相似度决定为文本分配给TTP标签，从而减少了仅仅在大型标签空间上竞争的复杂性。为此，我们提出了一种具有有效的基于采样的学习比较机制的神经匹配架构，促进学习过程。

    Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning pr
    

