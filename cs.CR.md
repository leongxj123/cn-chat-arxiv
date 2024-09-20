# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective](https://arxiv.org/abs/2402.18607) | 本文从对抗性的角度研究了分享扩散模型可能存在的隐私和公平风险，特别是探讨了在一方使用私人数据训练模型后提供给另一方黑盒访问权限的情况。 |
| [^2] | [Subsampling is not Magic: Why Large Batch Sizes Work for Differentially Private Stochastic Optimisation](https://arxiv.org/abs/2402.03990) | 通过研究差分隐私随机梯度下降（DP-SGD）中的总梯度方差，我们发现大批次大小有助于减小則采樣引起的方差，从而提高优化效果。 |
| [^3] | [Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems.](http://arxiv.org/abs/2311.00207) | 本文提出了Magmaw，这是一种针对基于机器学习的无线通信系统进行模态不可知对抗攻击的黑盒攻击方法。它能够生成通用的对抗扰动，并引入了新的攻击目标。实验证实了其对现有防御方法的韧性。使用实时无线攻击平台进行了概念验证。 |
| [^4] | [Designing an attack-defense game: how to increase robustness of financial transaction models via a competition.](http://arxiv.org/abs/2308.11406) | 通过设计一款攻防游戏，我们研究了使用序列金融数据的神经网络模型的对抗攻击和防御的现状和动态，并且通过分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。 |
| [^5] | [Visualising Personal Data Flows: Insights from a Case Study of Booking.com.](http://arxiv.org/abs/2304.09603) | 本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。 |

# 详细

[^1]: 在分享扩散模型中探讨隐私和公平风险：一种对抗性视角

    Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective

    [https://arxiv.org/abs/2402.18607](https://arxiv.org/abs/2402.18607)

    本文从对抗性的角度研究了分享扩散模型可能存在的隐私和公平风险，特别是探讨了在一方使用私人数据训练模型后提供给另一方黑盒访问权限的情况。

    

    扩散模型近年来在学术界和工业界引起了广泛关注，因为其在采样质量和分布覆盖方面表现出色。因此，提出了跨不同组织分享预训练扩散模型的建议，以提高数据利用率同时通过避免直接分享私人数据来增强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面调查。本文从对抗性的角度探讨了与分享扩散模型相关的潜在隐私和公平风险。具体而言，我们调查了一方（分享者）使用私人数据训练扩散模型并向另一方（接收者）提供预训练模型的黑盒访问权限用于下游任务的情况。我们展示了分享者可以实行的行动

    arXiv:2402.18607v1 Announce Type: cross  Abstract: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execut
    
[^2]: 則采樣并不是魔法: 大批量大小為什麼適用於差分隱私隨機優化

    Subsampling is not Magic: Why Large Batch Sizes Work for Differentially Private Stochastic Optimisation

    [https://arxiv.org/abs/2402.03990](https://arxiv.org/abs/2402.03990)

    通过研究差分隐私随机梯度下降（DP-SGD）中的总梯度方差，我们发现大批次大小有助于减小則采樣引起的方差，从而提高优化效果。

    

    我們研究了批次大小對差分隱私隨機梯度下降（DP-SGD）中總梯度方差的影響，尋求對大批次大小有用性的理論解釋。由於DP-SGD是現代差分隱私深度學習的基礎，其性質已被廣泛研究，最近的工作在實踐中發現大批次大小有益。然而，對於這種好處的理論解釋目前最多只能說是啟發式的。我們首先觀察到，在DP-SGD中，總梯度方差可以分解為由則采樣和噪聲引起的方差。然後，我們證明在無限次迭代的極限情況下，有效的噪聲引起的方差對批次大小是不變的。剩下的則采樣引起的方差隨著批次大小的增大而減小，因此大批次大小減小了有效的總梯度方差。我們在數值上確認這種漸進的情況在實際環境中是相關的，當批次大小不小的時候會起作用，並且發現

    We study the effect of the batch size to the total gradient variance in differentially private stochastic gradient descent (DP-SGD), seeking a theoretical explanation for the usefulness of large batch sizes. As DP-SGD is the basis of modern DP deep learning, its properties have been widely studied, and recent works have empirically found large batch sizes to be beneficial. However, theoretical explanations of this benefit are currently heuristic at best. We first observe that the total gradient variance in DP-SGD can be decomposed into subsampling-induced and noise-induced variances. We then prove that in the limit of an infinite number of iterations, the effective noise-induced variance is invariant to the batch size. The remaining subsampling-induced variance decreases with larger batch sizes, so large batches reduce the effective total gradient variance. We confirm numerically that the asymptotic regime is relevant in practical settings when the batch size is not small, and find tha
    
[^3]: Magmaw: 对基于机器学习的无线通信系统的模态不可知对抗攻击

    Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems. (arXiv:2311.00207v1 [cs.CR])

    [http://arxiv.org/abs/2311.00207](http://arxiv.org/abs/2311.00207)

    本文提出了Magmaw，这是一种针对基于机器学习的无线通信系统进行模态不可知对抗攻击的黑盒攻击方法。它能够生成通用的对抗扰动，并引入了新的攻击目标。实验证实了其对现有防御方法的韧性。使用实时无线攻击平台进行了概念验证。

    

    机器学习在合并端到端无线通信系统的所有物理层模块以实现联合收发器优化方面发挥了重要作用。尽管已经有许多针对基于机器学习的无线系统的对抗攻击方法，但现有方法并未提供包括源数据的多模态、共同的物理层组件和无线领域约束在内的全面视角。本文提出了Magmaw，这是一种能够针对通过无线信道传输的任何多模态信号生成通用对抗扰动的黑盒攻击方法。我们进一步对基于机器学习的下游应用的对抗攻击引入了新的目标。实验证实了该攻击对现有广泛使用的对抗训练和扰动信号减法防御方法的韧性。为了概念证明，我们使用软件定义无线电系统构建了一个实时无线攻击平台。

    Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experi
    
[^4]: 设计一款攻防游戏：通过竞争来增加金融交易模型的鲁棒性

    Designing an attack-defense game: how to increase robustness of financial transaction models via a competition. (arXiv:2308.11406v1 [cs.LG])

    [http://arxiv.org/abs/2308.11406](http://arxiv.org/abs/2308.11406)

    通过设计一款攻防游戏，我们研究了使用序列金融数据的神经网络模型的对抗攻击和防御的现状和动态，并且通过分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。

    

    鉴于金融领域恶意攻击风险不断升级和由此引发的严重损害，对机器学习模型的对抗策略和鲁棒的防御机制有深入的理解至关重要。随着银行日益广泛采用更精确但潜在脆弱的神经网络，这一威胁变得更加严重。我们旨在调查使用序列金融数据作为输入的神经网络模型的对抗攻击和防御的当前状态和动态。为了实现这一目标，我们设计了一个比赛，允许对现代金融交易数据中的问题进行逼真而详细的研究。参与者直接竞争，因此可能的攻击和防御在接近真实条件下进行了检验。我们的主要贡献是分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。

    Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.  To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break i
    
[^5]: 可视化个人数据流：以Booking.com为例的案例研究

    Visualising Personal Data Flows: Insights from a Case Study of Booking.com. (arXiv:2304.09603v1 [cs.CR])

    [http://arxiv.org/abs/2304.09603](http://arxiv.org/abs/2304.09603)

    本文以Booking.com为基础，以可视化个人数据流为研究，展示公司如何分享消费者个人数据，并讨论使用隐私政策告知客户个人数据流的挑战和限制。本案例研究为未来更以数据流为导向的隐私政策分析和建立更全面的个人数据流本体论的研究提供了参考。

    

    商业机构持有和处理的个人数据量越来越多。政策和法律不断变化，要求这些公司在收集、存储、处理和共享这些数据方面更加透明。本文报告了我们以Booking.com为案例研究，从他们的隐私政策中提取个人数据流的可视化工作。通过展示该公司如何分享其消费者的个人数据，我们提出了问题，并扩展了有关使用隐私政策告知客户个人数据流范围的挑战和限制的讨论。更重要的是，本案例研究可以为未来更以数据流为导向的隐私政策分析和在复杂商业生态系统中建立更全面的个人数据流本体论的研究提供参考。

    Commercial organisations are holding and processing an ever-increasing amount of personal data. Policies and laws are continually changing to require these companies to be more transparent regarding collection, storage, processing and sharing of this data. This paper reports our work of taking Booking.com as a case study to visualise personal data flows extracted from their privacy policy. By showcasing how the company shares its consumers' personal data, we raise questions and extend discussions on the challenges and limitations of using privacy policy to inform customers the true scale and landscape of personal data flows. More importantly, this case study can inform us about future research on more data flow-oriented privacy policy analysis and on the construction of a more comprehensive ontology on personal data flows in complicated business ecosystems.
    

