# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing](https://arxiv.org/abs/2403.12399) | 本论文提出了用于社区拉票的动态多步对抗性攻击，使得对手能够利用基于梯度的攻击来策划目标选民的操纵。 |
| [^2] | [DP-SGD for non-decomposable objective functions.](http://arxiv.org/abs/2310.03104) | 本论文提出了一种针对非可分的目标函数的DP-SGD方法，解决了使用差分隐私进行训练时，相似性损失函数的$L_2$敏感度增长随着批量大小增加的问题。 |
| [^3] | [SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks.](http://arxiv.org/abs/2308.11845) | 本论文提出了SEA，一种用于归因基于查询的黑盒攻击的机器学习安全系统，通过利用隐藏马尔可夫模型框架来理解攻击的演变过程，并有效归因攻击，即使是对于第二次出现的攻击，具有鲁棒性，旨在实现取证和人类可解释的情报共享。 |
| [^4] | [UniASM: Binary Code Similarity Detection without Fine-tuning.](http://arxiv.org/abs/2211.01144) | 提出了一种新的二进制代码嵌入模型UniASM，并设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，提出了一种新的二进制函数tokenization方法，缓解了词汇外的问题，并通过消融实验得到了一些新的有价值的发现，实验证明UniASM优于其他模型。 |

# 详细

[^1]: 将网络选举化：用于社区拉票的动态多步对抗性攻击

    Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing

    [https://arxiv.org/abs/2403.12399](https://arxiv.org/abs/2403.12399)

    本论文提出了用于社区拉票的动态多步对抗性攻击，使得对手能够利用基于梯度的攻击来策划目标选民的操纵。

    

    在今天的世界中，对于社区拉票的在线社交网络操纵问题是一个真正关注的问题。受选民模型、网络上的观点和极化动态的研究启发，我们将社区拉票建模为一个通过对GNN进行基于梯度的攻击而在网络上进行的动态过程。现有的GNN攻击都是单步的，没有考虑网络中信息传播的动态级联特性。我们考虑了一个现实的场景，即对手使用GNN作为代理来预测和操纵选民偏好，特别是不确定的选民。对GNN的基于梯度的攻击通知对手可以进行战略操纵，以使得目标选民入教。具体而言，我们探讨了$\textit{社区拉票的最小预算攻击}$（MBACC）。我们证明了MBACC问题是NP困难的，并提出了动态多步对抗性社区拉票（MAC）来解决这一问题。MAC m

    arXiv:2403.12399v1 Announce Type: new  Abstract: The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore $\textit{minimum budget attacks for community canvassing}$ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC m
    
[^2]: 非可分的目标函数的DP-SGD方法

    DP-SGD for non-decomposable objective functions. (arXiv:2310.03104v1 [cs.LG])

    [http://arxiv.org/abs/2310.03104](http://arxiv.org/abs/2310.03104)

    本论文提出了一种针对非可分的目标函数的DP-SGD方法，解决了使用差分隐私进行训练时，相似性损失函数的$L_2$敏感度增长随着批量大小增加的问题。

    

    无监督预训练是开发计算机视觉模型和大型语言模型的常见步骤。在这种情况下，由于缺少标签，需要使用基于相似性的损失函数，如对比损失，来优化相似输入之间的距离并最大化不同输入之间的距离。随着隐私问题的增多，使用差分隐私来训练这些模型变得更加重要。然而，由于这些损失函数生成输入的方式，它们的$L_2$敏感度会随着批量大小的增加而增加，这对于差分隐私训练方法（如DP-SGD）特别不利。为了解决这个问题，我们开发了一种新的DP-SGD变体，用于基于相似性的损失函数，特别是常用的对比损失，通过一种新颖的方式处理目标函数的梯度，使得梯度的敏感度对于批量大小是$O(1)$。

    Unsupervised pre-training is a common step in developing computer vision models and large language models. In this setting, the absence of labels requires the use of similarity-based loss functions, such as contrastive loss, that favor minimizing the distance between similar inputs and maximizing the distance between distinct inputs. As privacy concerns mount, training these models using differential privacy has become more important. However, due to how inputs are generated for these losses, one of their undesirable properties is that their $L_2$ sensitivity can grow with increasing batch size. This property is particularly disadvantageous for differentially private training methods, such as DP-SGD. To overcome this issue, we develop a new DP-SGD variant for similarity based loss functions -- in particular the commonly used contrastive loss -- that manipulates gradients of the objective function in a novel way to obtain a senstivity of the summed gradient that is $O(1)$ for batch size
    
[^3]: SEA：可共享和可解释的基于查询的黑盒攻击归因

    SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks. (arXiv:2308.11845v1 [cs.LG])

    [http://arxiv.org/abs/2308.11845](http://arxiv.org/abs/2308.11845)

    本论文提出了SEA，一种用于归因基于查询的黑盒攻击的机器学习安全系统，通过利用隐藏马尔可夫模型框架来理解攻击的演变过程，并有效归因攻击，即使是对于第二次出现的攻击，具有鲁棒性，旨在实现取证和人类可解释的情报共享。

    

    机器学习系统容易受到来自基于查询的黑盒攻击的敌对样本的攻击。尽管有各种努力来检测和防止这些攻击，但仍然需要一种更全面的方法来记录、分析和分享攻击证据。虽然经典安全领域受益于成熟的取证和情报共享技术，但机器学习领域尚未找到一种方式来对攻击者进行画像，并分享关于他们的信息。为此，本论文引入了SEA，一种新颖的机器学习安全系统，用于为取证目的表征对机器学习系统的黑盒攻击，并促进可解释的情报共享。SEA利用隐藏马尔可夫模型框架将观察到的查询序列归因于已知的攻击，因此它能够理解攻击的演变过程而不仅仅关注最终的敌对样本。我们的评估结果显示，SEA能够有效进行攻击归因，即使是对于第二次出现的攻击，也具有鲁棒性。

    Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, there is a need for a more comprehensive approach to logging, analyzing, and sharing evidence of attacks. While classic security benefits from well-established forensics and intelligence sharing, Machine Learning is yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages the Hidden Markov Models framework to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than just focusing on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on their second occurrence, and is robus
    
[^4]: UniASM：无需微调的二进制代码相似性检测

    UniASM: Binary Code Similarity Detection without Fine-tuning. (arXiv:2211.01144v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.01144](http://arxiv.org/abs/2211.01144)

    提出了一种新的二进制代码嵌入模型UniASM，并设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，提出了一种新的二进制函数tokenization方法，缓解了词汇外的问题，并通过消融实验得到了一些新的有价值的发现，实验证明UniASM优于其他模型。

    

    二进制代码相似性检测被广泛用于各种二进制分析任务，如漏洞搜索、恶意软件检测、克隆检测和补丁分析。最近的研究表明，基于学习的二进制代码嵌入模型比传统的基于特征的方法更好。本文提出了一种新的基于transformer的二进制代码嵌入模型UniASM，用于学习二进制函数的表示。我们设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，我们提出了一种新的二进制函数tokenization方法，增加了tokens的语义信息并缓解了词汇外的问题。通过消融实验进行了深入分析，得到了一些新的有价值的发现，实验证明UniASM优于其他模型。

    Binary code similarity detection (BCSD) is widely used in various binary analysis tasks such as vulnerability search, malware detection, clone detection, and patch analysis. Recent studies have shown that the learning-based binary code embedding models perform better than the traditional feature-based approaches. In this paper, we propose a novel transformer-based binary code embedding model named UniASM to learn representations of the binary functions. We design two new training tasks to make the spatial distribution of the generated vectors more uniform, which can be used directly in BCSD without any fine-tuning. In addition, we present a new tokenization approach for binary functions, which increases the token's semantic information and mitigates the out-of-vocabulary (OOV) problem. We conduct an in-depth analysis of the factors affecting model performance through ablation experiments and obtain some new and valuable findings. The experimental results show that UniASM outperforms th
    

