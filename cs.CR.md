# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem](https://arxiv.org/abs/2403.03593) | 介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入 |
| [^2] | [Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective](https://arxiv.org/abs/2402.10686) | 通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险 |

# 详细

[^1]: 您信任您的模型吗？深度学习生态系统中新兴的恶意软件威胁

    Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem

    [https://arxiv.org/abs/2403.03593](https://arxiv.org/abs/2403.03593)

    介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入

    

    训练高质量的深度学习模型是一项具有挑战性的任务，这是因为需要计算和技术要求。越来越多的个人、机构和公司越来越多地依赖于在公共代码库中提供的预训练的第三方模型。这些模型通常直接使用或集成到产品管道中而没有特殊的预防措施，因为它们实际上只是以张量形式的数据，被认为是安全的。在本文中，我们提出了一种针对神经网络的新的机器学习供应链威胁。我们介绍了MaleficNet 2.0，一种在神经网络中嵌入自解压自执行恶意软件的新技术。MaleficNet 2.0使用扩频信道编码结合纠错技术在深度神经网络的参数中注入恶意有效载荷。MaleficNet 2.0注入技术具有隐蔽性，不会降低模型的性能，并且对...

    arXiv:2403.03593v1 Announce Type: cross  Abstract: Training high-quality deep learning models is a challenging task due to computational and technical requirements. A growing number of individuals, institutions, and companies increasingly rely on pre-trained, third-party models made available in public repositories. These models are often used directly or integrated in product pipelines with no particular precautions, since they are effectively just data in tensor form and considered safe. In this paper, we raise awareness of a new machine learning supply chain threat targeting neural networks. We introduce MaleficNet 2.0, a novel technique to embed self-extracting, self-executing malware in neural networks. MaleficNet 2.0 uses spread-spectrum channel coding combined with error correction techniques to inject malicious payloads in the parameters of deep neural networks. MaleficNet 2.0 injection technique is stealthy, does not degrade the performance of the model, and is robust against 
    
[^2]: 不确定性、校准和成员推理攻击：信息论视角

    Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective

    [https://arxiv.org/abs/2402.10686](https://arxiv.org/abs/2402.10686)

    通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险

    

    在成员推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否被用于训练目标模型。在本文中，我们在一个信息理论框架内分析了最先进的似然比攻击（LiRA）的性能，这个框架可以允许研究真实数据生成过程中的不确定性的影响，由有限训练数据集引起的认知不确定性以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收到的信息逐渐减少：置信向量（CV）披露，其中输出概率向量被发布；真实标签置信度（TLC）披露，其中只有模型分配给真实标签的概率是可用的；以及决策集（DS）披露。

    arXiv:2402.10686v1 Announce Type: cross  Abstract: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in 
    

