# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Whispers in the Machine: Confidentiality in LLM-integrated Systems](https://arxiv.org/abs/2402.06922) | 本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。 |
| [^2] | [DeSparsify: Adversarial Attack Against Token Sparsification Mechanisms in Vision Transformers](https://arxiv.org/abs/2402.02554) | 本文提出了一种对抗攻击方法DeSparsify，针对使用Token稀疏化机制的视觉Transformer，通过精心制作的对抗样本欺骗稀疏化机制，导致最坏情况的性能，以此耗尽操作系统的资源并保持隐蔽性。 |
| [^3] | [DISTINQT: A Distributed Privacy Aware Learning Framework for QoS Prediction for Future Mobile and Wireless Networks.](http://arxiv.org/abs/2401.10158) | DISTINQT是一种面向未来移动和无线网络的隐私感知分布式学习框架，用于QoS预测。 |
| [^4] | [A Black-box NLP Classifier Attacker.](http://arxiv.org/abs/2112.11660) | 本文提出了一个黑盒NLP分类器攻击模型，通过基于自注意机制的词选择和贪婪搜索算法进行词替换，解决了影响NLP领域传统图像攻击方法不适用的问题。 |

# 详细

[^1]: 机器中的私语：LLM集成系统中的保密性

    Whispers in the Machine: Confidentiality in LLM-integrated Systems

    [https://arxiv.org/abs/2402.06922](https://arxiv.org/abs/2402.06922)

    本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。

    

    大规模语言模型（LLM）越来越多地与外部工具集成。尽管这些集成可以显著提高LLM的功能，但它们也在不同组件之间创建了一个新的攻击面，可能泄露机密数据。具体而言，恶意工具可以利用LLM本身的漏洞来操纵模型并损害其他服务的数据，这引发了在LLM集成环境中如何保护私密数据的问题。在这项工作中，我们提供了一种系统评估LLM集成系统保密性的方法。为此，我们形式化了一个"秘密密钥"游戏，可以捕捉模型隐藏私人信息的能力。这使我们能够比较模型对保密性攻击的脆弱性以及不同防御策略的有效性。在这个框架中，我们评估了八种先前发表的攻击和四种防御方法。我们发现当前的防御方法缺乏泛化性能。

    Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization
    
[^2]: DeSparsify：对视觉Transformer中的Token稀疏化机制进行的对抗攻击

    DeSparsify: Adversarial Attack Against Token Sparsification Mechanisms in Vision Transformers

    [https://arxiv.org/abs/2402.02554](https://arxiv.org/abs/2402.02554)

    本文提出了一种对抗攻击方法DeSparsify，针对使用Token稀疏化机制的视觉Transformer，通过精心制作的对抗样本欺骗稀疏化机制，导致最坏情况的性能，以此耗尽操作系统的资源并保持隐蔽性。

    

    视觉Transformer在计算机视觉领域做出了巨大贡献，展现出在各种任务（如图像分类、目标检测）中的最先进性能。然而，它们的高计算要求随使用的Token数量呈二次增长。为解决这个问题，提出了Token稀疏化技术。这些技术采用了一种依赖输入的策略，将无关的Token从计算流程中丢弃，提高模型的效率。然而，它们的动态性和平均情况假设使它们容易受到一种新的威胁 - 经过精心制作的对抗样本，能够欺骗稀疏化机制，导致最坏情况的性能。在本文中，我们提出了一种攻击方法DeSparsify，针对使用Token稀疏化机制的视觉Transformer的可用性。该攻击旨在耗尽操作系统的资源，同时保持隐蔽性。

    Vision transformers have contributed greatly to advancements in the computer vision domain, demonstrating state-of-the-art performance in diverse tasks (e.g., image classification, object detection). However, their high computational requirements grow quadratically with the number of tokens used. Token sparsification techniques have been proposed to address this issue. These techniques employ an input-dependent strategy, in which uninformative tokens are discarded from the computation pipeline, improving the model's efficiency. However, their dynamism and average-case assumption makes them vulnerable to a new threat vector - carefully crafted adversarial examples capable of fooling the sparsification mechanism, resulting in worst-case performance. In this paper, we present DeSparsify, an attack targeting the availability of vision transformers that use token sparsification mechanisms. The attack aims to exhaust the operating system's resources, while maintaining its stealthiness. Our e
    
[^3]: DISTINQT: 一种面向未来移动和无线网络的分布式隐私感知学习框架，用于QoS预测

    DISTINQT: A Distributed Privacy Aware Learning Framework for QoS Prediction for Future Mobile and Wireless Networks. (arXiv:2401.10158v1 [cs.NI])

    [http://arxiv.org/abs/2401.10158](http://arxiv.org/abs/2401.10158)

    DISTINQT是一种面向未来移动和无线网络的隐私感知分布式学习框架，用于QoS预测。

    

    5G和6G以后的网络将支持依赖一定服务质量（QoS）的新的和具有挑战性的用例和应用程序。及时预测QoS对于安全关键应用（如车辆通信）尤为重要。尽管直到最近，QoS预测一直由集中式人工智能（AI）解决方案完成，但已经出现了一些隐私、计算和运营方面的问题。替代方案已经出现（如分割学习、联邦学习），将复杂度较低的AI任务分布在节点之间，同时保护数据隐私。然而，考虑到未来无线网络的异构性，当涉及可扩展的分布式学习方法时，会出现新的挑战。该研究提出了一种名为DISTINQT的面向QoS预测的隐私感知分布式学习框架。

    Beyond 5G and 6G networks are expected to support new and challenging use cases and applications that depend on a certain level of Quality of Service (QoS) to operate smoothly. Predicting the QoS in a timely manner is of high importance, especially for safety-critical applications as in the case of vehicular communications. Although until recent years the QoS prediction has been carried out by centralized Artificial Intelligence (AI) solutions, a number of privacy, computational, and operational concerns have emerged. Alternative solutions have been surfaced (e.g. Split Learning, Federated Learning), distributing AI tasks of reduced complexity across nodes, while preserving the privacy of the data. However, new challenges rise when it comes to scalable distributed learning approaches, taking into account the heterogeneous nature of future wireless networks. The current work proposes DISTINQT, a privacy-aware distributed learning framework for QoS prediction. Our framework supports mult
    
[^4]: 一个黑盒NLP分类器攻击器

    A Black-box NLP Classifier Attacker. (arXiv:2112.11660v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.11660](http://arxiv.org/abs/2112.11660)

    本文提出了一个黑盒NLP分类器攻击模型，通过基于自注意机制的词选择和贪婪搜索算法进行词替换，解决了影响NLP领域传统图像攻击方法不适用的问题。

    

    深度神经网络在解决各种现实世界任务中具有广泛的应用，并在计算机视觉、图像分类和自然语言处理等领域取得了令人满意的结果。与此同时，神经网络的安全性和鲁棒性已经变得非常重要，因为各种研究已经显示了神经网络的脆弱性。以自然语言处理任务为例，神经网络可能被一个与原始文本高度相似的、经过仔细修改的文本所迷惑。根据之前的研究，大部分研究都集中在图像领域；与图像对抗攻击不同，文本以离散序列表示，传统的图像攻击方法在NLP领域不适用。本文提出了一个基于自注意机制的词级NLP情感分类器攻击模型，其中包括基于词选择的自注意机制和贪婪搜索算法进行词替换。我们进行了实验验证...

    Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experimen
    

