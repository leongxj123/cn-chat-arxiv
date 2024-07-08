# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Blockchain-empowered Federated Learning: Benefits, Challenges, and Solutions](https://arxiv.org/abs/2403.00873) | 区块链技术被整合到联邦学习系统中以提供更强的安全性、公平性和可扩展性，但也引入了额外的网络、计算和存储资源需求。 |
| [^2] | [Defending Jailbreak Prompts via In-Context Adversarial Game](https://arxiv.org/abs/2402.13148) | 介绍了一种通过上下文对抗游戏(ICAG)防御越狱提示的方法，能够显著降低成功率。 |
| [^3] | [Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack.](http://arxiv.org/abs/2401.09673) | 本文提出了一种名为本地自适应对抗颜色攻击（LAACA）的方法，用于保护艺术品免受神经风格转换（NST）的滥用。该方法通过在不可察觉的情况下对图像进行修改，产生对NST具有干扰作用的扰动。 |
| [^4] | [Malla: Demystifying Real-world Large Language Model Integrated Malicious Services.](http://arxiv.org/abs/2401.03315) | 本研究对212个真实的恶意服务（Malla）进行了系统研究，揭示了它们在地下市场的扩散和对公共LLM服务的影响，以及其使用的策略和技术。 |
| [^5] | [Smooth Lower Bounds for Differentially Private Algorithms via Padding-and-Permuting Fingerprinting Codes.](http://arxiv.org/abs/2307.07604) | 本论文提出了一种通过填充和置换指纹编码的方法来产生困难实例，从而在各种情景下提供平滑下界。这方法适用于差分隐私平均问题和近似k. |
| [^6] | [Certifiable Black-Box Attack: Ensuring Provably Successful Attack for Adversarial Examples.](http://arxiv.org/abs/2304.04343) | 本文提出可证黑盒攻击，能够保证攻击成功率，设计了多种新技术。 |

# 详细

[^1]: 区块链赋能的联邦学习: 好处、挑战和解决方案

    Blockchain-empowered Federated Learning: Benefits, Challenges, and Solutions

    [https://arxiv.org/abs/2403.00873](https://arxiv.org/abs/2403.00873)

    区块链技术被整合到联邦学习系统中以提供更强的安全性、公平性和可扩展性，但也引入了额外的网络、计算和存储资源需求。

    

    联邦学习(FL)是一种分布式机器学习方法，通过在客户端本地训练模型并在参数服务器上进行聚合来保护用户数据隐私。尽管在保护隐私方面有效，但FL系统面临单点故障、缺乏激励和不足的安全性等局限性。为了应对这些挑战，将区块链技术整合到FL系统中，以提供更强的安全性、公平性和可扩展性。然而，区块链赋能的FL(BC-FL)系统对网络、计算和存储资源提出了额外的需求。本调查全面审查了最近关于BC-FL系统的研究，分析了与区块链整合相关的好处和挑战。我们探讨了区块链为何适用于FL，如何实施以及整合的挑战和现有解决方案。此外，我们还提供了关于未来研究方向的见解。

    arXiv:2403.00873v1 Announce Type: cross  Abstract: Federated learning (FL) is a distributed machine learning approach that protects user data privacy by training models locally on clients and aggregating them on a parameter server. While effective at preserving privacy, FL systems face limitations such as single points of failure, lack of incentives, and inadequate security. To address these challenges, blockchain technology is integrated into FL systems to provide stronger security, fairness, and scalability. However, blockchain-empowered FL (BC-FL) systems introduce additional demands on network, computing, and storage resources. This survey provides a comprehensive review of recent research on BC-FL systems, analyzing the benefits and challenges associated with blockchain integration. We explore why blockchain is applicable to FL, how it can be implemented, and the challenges and existing solutions for its integration. Additionally, we offer insights on future research directions fo
    
[^2]: 通过上下文对抗游戏防御越狱提示

    Defending Jailbreak Prompts via In-Context Adversarial Game

    [https://arxiv.org/abs/2402.13148](https://arxiv.org/abs/2402.13148)

    介绍了一种通过上下文对抗游戏(ICAG)防御越狱提示的方法，能够显著降低成功率。

    

    大语言模型(LLMs)展现出在不同应用领域中的显著能力。然而，对其安全性的担忧，特别是对越狱攻击的脆弱性，仍然存在。受到深度学习中对抗训练和LLM代理学习过程的启发，我们引入了上下文对抗游戏(ICAG)来防御越狱攻击，无需进行微调。ICAG利用代理学习进行对抗游戏，旨在动态扩展知识以防御越狱攻击。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一持续改进过程加强了对新生成的越狱提示的防御。我们的实证研究证实了ICAG的有效性，经由ICAG保护的LLMs在各种攻击场景中显著降低了越狱成功率。

    arXiv:2402.13148v1 Announce Type: new  Abstract: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Mo
    
[^3]: 使用本地自适应对抗颜色攻击对艺术品进行神经风格转换的保护

    Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack. (arXiv:2401.09673v1 [cs.CV])

    [http://arxiv.org/abs/2401.09673](http://arxiv.org/abs/2401.09673)

    本文提出了一种名为本地自适应对抗颜色攻击（LAACA）的方法，用于保护艺术品免受神经风格转换（NST）的滥用。该方法通过在不可察觉的情况下对图像进行修改，产生对NST具有干扰作用的扰动。

    

    神经风格转换（NST）广泛应用于计算机视觉中，用于生成具有任意风格的新图像。这个过程利用神经网络将风格图像的美学元素与内容图像的结构因素融合在一起，形成一个和谐整合的视觉结果。然而，未经授权的NST可能会滥用艺术品。这种滥用引起了关于艺术家权利的社会技术问题，并促使开发技术方法来积极保护原始创作。对抗性攻击主要在机器学习安全中进行探索。我们的工作将这一技术引入到保护艺术家知识产权的领域。本文引入了本地自适应对抗颜色攻击（LAACA）的方法，这种方法可以以对人眼不可察觉但对NST产生干扰的方式修改图像。具体而言，我们设计了针对高频内容丰富区域的扰动，这些扰动由中间特征的破坏产生。我们进行了实验和用户研究。

    Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study
    
[^4]: Malla: 揭秘现实世界中大规模语言模型整合恶意服务

    Malla: Demystifying Real-world Large Language Model Integrated Malicious Services. (arXiv:2401.03315v1 [cs.CR])

    [http://arxiv.org/abs/2401.03315](http://arxiv.org/abs/2401.03315)

    本研究对212个真实的恶意服务（Malla）进行了系统研究，揭示了它们在地下市场的扩散和对公共LLM服务的影响，以及其使用的策略和技术。

    

    大规模语言模型（LLMs）的地下利用，也称为Malla，正在增加，加剧了网络安全威胁，并对LLMs技术的可信度提出了疑问。然而，迄今为止，很少有工作努力去了解这种新型网络犯罪的规模、影响和技术。本文是第一次对212个真实的Malla进行系统研究，揭示了它们在地下市场的扩散，并揭示了它们的操作模式。我们的研究揭开了Malla生态系统，揭示了其显著的增长对当今公共LLM服务的影响。通过对212个Mallas进行研究，我们发现了8个后端LLMs，以及182个绕过公共LLM API保护措施的提示。我们进一步揭示了Mallas使用的策略，包括滥用未经审查的LLMs和通过越狱提示利用公共LLM API。我们的发现有助于更好地理解Malla犯罪行为的实质。

    The underground exploitation of large language models (LLMs) for malicious services (i.e., Malla) is witnessing an uptick, amplifying the cyber threat landscape and posing questions about the trustworthiness of LLM technologies. However, there has been little effort to understand this new cybercrime, in terms of its magnitude, impact, and techniques. In this paper, we conduct the first systematic study on 212 real-world Mallas, uncovering their proliferation in underground marketplaces and exposing their operational modalities. Our study discloses the Malla ecosystem, revealing its significant growth and impact on today's public LLM services. Through examining 212 Mallas, we uncovered eight backend LLMs used by Mallas, along with 182 prompts that circumvent the protective measures of public LLM APIs. We further demystify the tactics employed by Mallas, including the abuse of uncensored LLMs and the exploitation of public LLM APIs through jailbreak prompts. Our findings enable a better 
    
[^5]: 通过填充和置换指纹编码的方法，对差分隐私算法提供平滑下界

    Smooth Lower Bounds for Differentially Private Algorithms via Padding-and-Permuting Fingerprinting Codes. (arXiv:2307.07604v1 [cs.CR])

    [http://arxiv.org/abs/2307.07604](http://arxiv.org/abs/2307.07604)

    本论文提出了一种通过填充和置换指纹编码的方法来产生困难实例，从而在各种情景下提供平滑下界。这方法适用于差分隐私平均问题和近似k.

    

    指纹编码方法是最广泛用于确定约束差分隐私算法的样本复杂度或错误率的方法。然而，对于许多差分隐私问题，我们并不知道适当的下界，并且即使对于我们知道的问题，下界也不平滑，并且通常在误差大于某个阈值时变得无意义。在这项工作中，我们通过将填充和置换转换应用于指纹编码，提出了一种生成困难实例的简单方法。我们通过在不同情景下提供新的下界来说明这种方法的适用性：1. 低准确度情景下差分隐私平均问题的紧密下界，这尤其意味着新的私有1簇问题的下界 2. 近似k

    Fingerprinting arguments, first introduced by Bun, Ullman, and Vadhan (STOC 2014), are the most widely used method for establishing lower bounds on the sample complexity or error of approximately differentially private (DP) algorithms. Still, there are many problems in differential privacy for which we don't know suitable lower bounds, and even for problems that we do, the lower bounds are not smooth, and usually become vacuous when the error is larger than some threshold.  In this work, we present a simple method to generate hard instances by applying a padding-and-permuting transformation to a fingerprinting code. We illustrate the applicability of this method by providing new lower bounds in various settings:  1. A tight lower bound for DP averaging in the low-accuracy regime, which in particular implies a new lower bound for the private 1-cluster problem introduced by Nissim, Stemmer, and Vadhan (PODS 2016).  2. A lower bound on the additive error of DP algorithms for approximate k
    
[^6]: 可证黑盒攻击：确保对抗性样本的攻击成功率

    Certifiable Black-Box Attack: Ensuring Provably Successful Attack for Adversarial Examples. (arXiv:2304.04343v1 [cs.LG])

    [http://arxiv.org/abs/2304.04343](http://arxiv.org/abs/2304.04343)

    本文提出可证黑盒攻击，能够保证攻击成功率，设计了多种新技术。

    

    黑盒对抗攻击具有破坏机器学习模型的强大潜力。现有的黑盒对抗攻击通过迭代查询目标模型和/或利用本地代理模型的可转移性来制作对抗样本。当实验设计攻击时，攻击是否成功对攻击者来说仍然是未知的。本文通过修改随机平滑性理论，首次研究了可证黑盒攻击的新范例，能够保证制作的对抗样本的攻击成功率，为此设计了多种新技术。

    Black-box adversarial attacks have shown strong potential to subvert machine learning models. Existing black-box adversarial attacks craft the adversarial examples by iteratively querying the target model and/or leveraging the transferability of a local surrogate model. Whether such attack can succeed remains unknown to the adversary when empirically designing the attack. In this paper, to our best knowledge, we take the first step to study a new paradigm of adversarial attacks -- certifiable black-box attack that can guarantee the attack success rate of the crafted adversarial examples. Specifically, we revise the randomized smoothing to establish novel theories for ensuring the attack success rate of the adversarial examples. To craft the adversarial examples with the certifiable attack success rate (CASR) guarantee, we design several novel techniques, including a randomized query method to query the target model, an initialization method with smoothed self-supervised perturbation to
    

