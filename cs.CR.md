# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SPEAR:Exact Gradient Inversion of Batches in Federated Learning](https://arxiv.org/abs/2403.03945) | 该论文提出了第一个能够精确重构批量$b >1$的算法，在联邦学习中解决了梯度反演攻击的问题。 |
| [^2] | [Query-Based Adversarial Prompt Generation](https://arxiv.org/abs/2402.12329) | 该研究提出了一种基于查询的对抗性攻击方法，通过利用远程语言模型的 API 访问构造对抗性示例，使模型以更高概率发出有害字符串，而非仅仅基于模型之间的转移性攻击。 |
| [^3] | [Explainable Adversarial Learning Framework on Physical Layer Secret Keys Combating Malicious Reconfigurable Intelligent Surface](https://arxiv.org/abs/2402.06663) | 本文提出了一个对抗学习框架，用于合法参与方间的物理层密钥生成，在恶意可重构智能面干扰下提供了一个可解释的解决方案。 |
| [^4] | [Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning.](http://arxiv.org/abs/2306.09273) | 这篇论文提出了一种针对值函数算法和梯度算法的攻击方法，利用梯度反转重建状态、动作和监督信号，以解决嵌入式人工智能中的隐私泄露问题。 |
| [^5] | [Revisiting DeepFool: generalization and improvement.](http://arxiv.org/abs/2303.12481) | 本文提出了一种新的对抗性攻击，该攻击是广义了DeepFool攻击，既有效又计算效率高，适用于评估大型深度神经网络的鲁棒性。 |

# 详细

[^1]: SPEAR：联邦学习中批量精确梯度反演

    SPEAR:Exact Gradient Inversion of Batches in Federated Learning

    [https://arxiv.org/abs/2403.03945](https://arxiv.org/abs/2403.03945)

    该论文提出了第一个能够精确重构批量$b >1$的算法，在联邦学习中解决了梯度反演攻击的问题。

    

    联邦学习是一种流行的协作机器学习框架，在这个框架中，多个客户端仅与服务器共享他们本地数据的梯度更新，而不是实际数据。不幸的是，最近发现梯度反演攻击可以从这些共享的梯度中重构出数据。现有的攻击只能在重要的诚实但好奇设置中对批量大小为$b=1$的数据进行精确重构，对于更大的批量只能进行近似重构。在这项工作中，我们提出了\emph{第一个准确重建批量$b >1$的算法}。这种方法结合了对梯度显式低秩结构的数学见解和基于采样的算法。关键的是，我们利用ReLU诱导的梯度稀疏性，精确地过滤掉大量错误的样本，使最终的重建步骤可行。我们为全连接提供了高效的GPU实现

    arXiv:2403.03945v1 Announce Type: new  Abstract: Federated learning is a popular framework for collaborative machine learning where multiple clients only share gradient updates on their local data with the server and not the actual data. Unfortunately, it was recently shown that gradient inversion attacks can reconstruct this data from these shared gradients. Existing attacks enable exact reconstruction only for a batch size of $b=1$ in the important honest-but-curious setting, with larger batches permitting only approximate reconstruction. In this work, we propose \emph{the first algorithm reconstructing whole batches with $b >1$ exactly}. This approach combines mathematical insights into the explicit low-rank structure of gradients with a sampling-based algorithm. Crucially, we leverage ReLU-induced gradient sparsity to precisely filter out large numbers of incorrect samples, making a final reconstruction step tractable. We provide an efficient GPU implementation for fully connected 
    
[^2]: 基于查询的对抗性提示生成

    Query-Based Adversarial Prompt Generation

    [https://arxiv.org/abs/2402.12329](https://arxiv.org/abs/2402.12329)

    该研究提出了一种基于查询的对抗性攻击方法，通过利用远程语言模型的 API 访问构造对抗性示例，使模型以更高概率发出有害字符串，而非仅仅基于模型之间的转移性攻击。

    

    最近的研究表明，可以构造对抗性示例，导致一个对其进行了调整的语言模型产生有害字符串或执行有害行为。现有的攻击要么在白盒设置中（完全访问模型权重），要么通过可转移性：一种现象，即在一个模型上精心设计的对抗性示例通常在其他模型上仍然有效。我们通过基于查询的攻击改进以前的工作，利用 API 访问远程语言模型来构造对抗性示例，使模型以（明显）更高的概率发出有害字符串，而不能仅仅使用转移攻击。我们在 GPT-3.5 和 OpenAI 的安全分类器上验证了我们的攻击；我们能够让 GPT-3.5 发出有害字符串，而目前的转移攻击失败了，并且我们几乎以 100% 的概率规避了安全分类器。

    arXiv:2402.12329v1 Announce Type: cross  Abstract: Recent work has shown it is possible to construct adversarial examples that cause an aligned language model to emit harmful strings or perform harmful behavior. Existing attacks work either in the white-box setting (with full access to the model weights), or through transferability: the phenomenon that adversarial examples crafted on one model often remain effective on other models. We improve on prior work with a query-based attack that leverages API access to a remote language model to construct adversarial examples that cause the model to emit harmful strings with (much) higher probability than with transfer-only attacks. We validate our attack on GPT-3.5 and OpenAI's safety classifier; we can cause GPT-3.5 to emit harmful strings that current transfer attacks fail at, and we can evade the safety classifier with nearly 100% probability.
    
[^3]: 物理层密钥对抗恶意可重构智能面的可解释对抗学习框架

    Explainable Adversarial Learning Framework on Physical Layer Secret Keys Combating Malicious Reconfigurable Intelligent Surface

    [https://arxiv.org/abs/2402.06663](https://arxiv.org/abs/2402.06663)

    本文提出了一个对抗学习框架，用于合法参与方间的物理层密钥生成，在恶意可重构智能面干扰下提供了一个可解释的解决方案。

    

    可重构智能面（RIS）的发展对物理层安全（PLS）是一把双刃剑。合法的RIS可以产生有益的影响，包括增加信道的随机性，增强物理层密钥生成（PL-SKG），而恶意的RIS可以破坏合法信道并破解大部分现有的PL-SKG。在这项工作中，我们提出了一个合法参与方（即爱丽丝和鲍勃）之间的对抗学习框架，以解决中间人恶意RIS（MITM-RIS）窃听问题。首先，我们推导了合法配对和MITM-RIS之间的理论互信息差距。然后，爱丽丝和鲍勃利用生成对抗网络（GAN）学习实现一个与MITM-RIS没有互信息重叠的共同特征面。接下来，我们使用符号可解释AI（xAI）表示对黑盒神经网络进行信号处理解释。这些主导神经元的符号术语有助于特征工程。

    The development of reconfigurable intelligent surfaces (RIS) is a double-edged sword to physical layer security (PLS). Whilst a legitimate RIS can yield beneficial impacts including increased channel randomness to enhance physical layer secret key generation (PL-SKG), malicious RIS can poison legitimate channels and crack most of existing PL-SKGs. In this work, we propose an adversarial learning framework between legitimate parties (namely Alice and Bob) to address this Man-in-the-middle malicious RIS (MITM-RIS) eavesdropping. First, the theoretical mutual information gap between legitimate pairs and MITM-RIS is deduced. Then, Alice and Bob leverage generative adversarial networks (GANs) to learn to achieve a common feature surface that does not have mutual information overlap with MITM-RIS. Next, we aid signal processing interpretation of black-box neural networks by using a symbolic explainable AI (xAI) representation. These symbolic terms of dominant neurons aid feature engineering-
    
[^4]: 你的房间不是私密的：关于强化学习的梯度反转攻击

    Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning. (arXiv:2306.09273v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2306.09273](http://arxiv.org/abs/2306.09273)

    这篇论文提出了一种针对值函数算法和梯度算法的攻击方法，利用梯度反转重建状态、动作和监督信号，以解决嵌入式人工智能中的隐私泄露问题。

    

    嵌入式人工智能的显著发展吸引了人们的极大关注，该技术使得机器人可以在虚拟环境中导航、感知和互动。由于计算机视觉和大型语言模型方面的显著进展，隐私问题在嵌入式人工智能领域变得至关重要，因为机器人可以访问大量个人信息。然而，关于强化学习算法中的隐私泄露问题，尤其是关于值函数算法和梯度算法的问题，在研究中尚未得到充分考虑。本文旨在通过提出一种攻击值函数算法和梯度算法的方法，利用梯度反转重建状态、动作和监督信号，来解决这一问题。选择使用梯度进行攻击是因为常用的联邦学习技术仅利用基于私人用户数据计算的梯度来优化模型，而不存储或传输用户数据。

    The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advancements in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly in relation to reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervision signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or trans
    
[^5]: 重新审视DeepFool：泛化和改进

    Revisiting DeepFool: generalization and improvement. (arXiv:2303.12481v1 [cs.LG])

    [http://arxiv.org/abs/2303.12481](http://arxiv.org/abs/2303.12481)

    本文提出了一种新的对抗性攻击，该攻击是广义了DeepFool攻击，既有效又计算效率高，适用于评估大型深度神经网络的鲁棒性。

    

    深度神经网络被已知容易受到对抗样本的攻击，这些输入稍加修改便会导致网络做出错误的预测。这导致了大量研究，以评估这些网络对此类扰动的鲁棒性度量。最小l2对抗扰动的鲁棒性，是一种特别重要的鲁棒性度量。然而，现有的用于评估此类鲁棒性度量的方法，要么计算成本高，要么不太准确。在本文中，我们引入了一种新的对抗性攻击方法，它在效果和计算效率之间保持平衡。我们提出的攻击是广义了深度欺骗（DeepFool）攻击，但它们仍然易于理解和实现。我们展示了我们的攻击在效果和计算效率方面均优于现有方法。我们提出的攻击也适用于评估大型深度神经网络的鲁棒性。

    Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal l2 adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large
    

