# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach](https://arxiv.org/abs/2402.02672) | 本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。 |
| [^2] | [Passive Inference Attacks on Split Learning via Adversarial Regularization.](http://arxiv.org/abs/2310.10483) | 该论文介绍了一种针对拆分学习的被动推理攻击框架SDAR，通过利用辅助数据和对抗性正则化来推断客户端的私有特征和标签，在实验中取得了与主动攻击相当的攻击性能。 |
| [^3] | [Optimal Differentially Private Learning with Public Data.](http://arxiv.org/abs/2306.15056) | 本论文研究了具有公共数据的最优差分隐私学习，并解决了在训练差分隐私模型时如何利用公共数据提高准确性的问题。 |
| [^4] | [DNN-Defender: An in-DRAM Deep Neural Network Defense Mechanism for Adversarial Weight Attack.](http://arxiv.org/abs/2305.08034) | DNN-Defender是一种基于DRAM的受害者重点防御机制，适用于量化DNN，利用内存中交换的潜力以抵御有针对性的位翻转攻击，可以提供高水平的保护。 |

# 详细

[^1]: 对分布式数据的条件平均治疗效果估计：一种保护隐私的方法

    Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach

    [https://arxiv.org/abs/2402.02672](https://arxiv.org/abs/2402.02672)

    本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。

    

    在医学和社会科学等各个领域中，对条件平均治疗效果（CATEs）的估计是一个重要的课题。如果分布在多个参与方之间的数据可以集中，可以对CATEs进行高精度的估计。然而，如果这些数据包含隐私信息，则很难进行数据聚合。为了解决这个问题，我们提出了数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计CATE模型，并通过数值实验对该方法进行了评估。我们的贡献总结如下三点。首先，我们的方法能够在分布式数据上进行非迭代通信的半参数CATE模型的估计和测试。半参数或非参数的CATE模型能够比参数模型更稳健地进行估计和测试，对于模型偏差的鲁棒性更强。然而，据我们所知，目前还没有提出有效的通信方法来估计和测试这些模型。

    Estimation of conditional average treatment effects (CATEs) is an important topic in various fields such as medical and social sciences. CATEs can be estimated with high accuracy if distributed data across multiple parties can be centralized. However, it is difficult to aggregate such data if they contain privacy information. To address this issue, we proposed data collaboration double machine learning (DC-DML), a method that can estimate CATE models with privacy preservation of distributed data, and evaluated the method through numerical experiments. Our contributions are summarized in the following three points. First, our method enables estimation and testing of semi-parametric CATE models without iterative communication on distributed data. Semi-parametric or non-parametric CATE models enable estimation and testing that is more robust to model mis-specification than parametric models. However, to our knowledge, no communication-efficient method has been proposed for estimating and 
    
[^2]: 通过对抗性正则化对拆分学习进行袭击的被动推理攻击

    Passive Inference Attacks on Split Learning via Adversarial Regularization. (arXiv:2310.10483v4 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2310.10483](http://arxiv.org/abs/2310.10483)

    该论文介绍了一种针对拆分学习的被动推理攻击框架SDAR，通过利用辅助数据和对抗性正则化来推断客户端的私有特征和标签，在实验中取得了与主动攻击相当的攻击性能。

    

    拆分学习(SL)已成为传统联邦学习的一种实用且高效的替代方案。虽然以前攻击SL的尝试往往依赖于过于强硬的假设或者针对易受攻击的模型，但我们试图开发更加实用的攻击方法。我们引入了SDAR，这是一个针对拥有诚实但好奇的服务器的SL的新攻击框架。SDAR利用辅助数据和对抗性正则化来学习客户端私有模型的可解码模拟器，在基本SL下可以有效地推断出客户端的私有特征，并在U型SL下推断出特征和标签。我们进行了大量实验来验证我们提出的攻击方法的有效性。值得注意的是，在具有挑战性但实际的场景中，现有的被动攻击难以有效地重建客户端的私有数据时，SDAR始终实现了与主动攻击相当的攻击性能。在CIFAR-10上，在深度拆分水平为7的情况下，SDAR达到了攻击性能。

    Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achi
    
[^3]: 具有公共数据的最优差分隐私学习

    Optimal Differentially Private Learning with Public Data. (arXiv:2306.15056v1 [cs.LG])

    [http://arxiv.org/abs/2306.15056](http://arxiv.org/abs/2306.15056)

    本论文研究了具有公共数据的最优差分隐私学习，并解决了在训练差分隐私模型时如何利用公共数据提高准确性的问题。

    

    差分隐私能够确保训练机器学习模型不泄漏私密数据。然而，差分隐私的代价是模型的准确性降低或样本复杂度增加。在实践中，我们可能可以访问不涉及隐私问题的辅助公共数据。这促使了最近研究公共数据在提高差分隐私模型准确性方面的作用。在本研究中，我们假设有一定数量的公共数据，并解决以下基本开放问题：1.在有公共数据的情况下，训练基于私有数据集的差分隐私模型的最优（最坏情况）误差是多少？哪些算法是最优的？2.如何利用公共数据在实践中改进差分隐私模型训练？我们在本地模型和中心模型的差分隐私问题下考虑这些问题。为了回答第一个问题，我们证明了对三个基本问题的最优误差率的紧密（最高常数因子）下界和上界。这三个问题是：均值估计，经验风险最小化和凸奇化。

    Differential Privacy (DP) ensures that training a machine learning model does not leak private data. However, the cost of DP is lower model accuracy or higher sample complexity. In practice, we may have access to auxiliary public data that is free of privacy concerns. This has motivated the recent study of what role public data might play in improving the accuracy of DP models. In this work, we assume access to a given amount of public data and settle the following fundamental open questions: 1. What is the optimal (worst-case) error of a DP model trained over a private data set while having access to side public data? What algorithms are optimal? 2. How can we harness public data to improve DP model training in practice? We consider these questions in both the local and central models of DP. To answer the first question, we prove tight (up to constant factors) lower and upper bounds that characterize the optimal error rates of three fundamental problems: mean estimation, empirical ris
    
[^4]: DNN-Defender: 一种用于对抗 Adversarial Weight Attack 的内存中深度神经网络防御机制

    DNN-Defender: An in-DRAM Deep Neural Network Defense Mechanism for Adversarial Weight Attack. (arXiv:2305.08034v1 [cs.CR])

    [http://arxiv.org/abs/2305.08034](http://arxiv.org/abs/2305.08034)

    DNN-Defender是一种基于DRAM的受害者重点防御机制，适用于量化DNN，利用内存中交换的潜力以抵御有针对性的位翻转攻击，可以提供高水平的保护。

    

    随着深度学习在许多安全敏感领域中的部署，机器学习安全变得越来越重要。最近的研究表明，攻击者可以利用DRAM的RowHammer漏洞，以确定性和精确性地翻转深度神经网络(DNN)模型权重的位，从而影响推断准确性。现有的防御机制是基于软件的，例如重构权重需要昂贵的训练开销或性能下降。另一方面，基于通用硬件的受害者/攻击者重点防御机制会导致昂贵的硬件开销，并保留受害者和攻击者行之间的空间连接。在本文中，我们提出了第一种针对量化DNN量身定制的基于DRAM的受害者重点防御机制，称为DNN-Defender，利用了内存中交换的潜力以抵御有针对性的位翻转攻击。我们的结果表明，DNN-Defender可以提供高水平的保护。

    With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks. Our results indicate that DNN-Defender can deliver a high level of protection dow
    

