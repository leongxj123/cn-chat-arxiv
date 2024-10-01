# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tastle: Distract Large Language Models for Automatic Jailbreak Attack](https://arxiv.org/abs/2403.08424) | Tastle是一种新颖的黑盒越狱框架，采用恶意内容隐藏和内存重构以及迭代优化算法，用于自动对大型语言模型进行红队攻击。 |
| [^2] | [WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off](https://arxiv.org/abs/2403.04808) | WaterMax提出了一种新的水印方案，能够在保持生成文本质量的同时实现高检测性能，打破了水印技术中质量和稳健性之间的传统平衡。 |
| [^3] | [The Art of Deception: Robust Backdoor Attack using Dynamic Stacking of Triggers.](http://arxiv.org/abs/2401.01537) | 这项研究介绍了一种使用动态触发器进行强健后门攻击的方法，通过巧妙设计的调整，使损坏的样本与干净的样本无法区分，实验证明这种方法可以成功地欺骗语音识别系统。 |
| [^4] | [CyberForce: A Federated Reinforcement Learning Framework for Malware Mitigation.](http://arxiv.org/abs/2308.05978) | CyberForce是一个联邦强化学习框架，用于在物联网设备中协同私密地确定适合缓解各种零日攻击的MTD技术。它整合了设备指纹识别和异常检测，并通过奖励或惩罚FRL agent选择的MTD机制来提高网络安全性。 |

# 详细

[^1]: Tastle: 为自动越狱攻击干扰大型语言模型

    Tastle: Distract Large Language Models for Automatic Jailbreak Attack

    [https://arxiv.org/abs/2403.08424](https://arxiv.org/abs/2403.08424)

    Tastle是一种新颖的黑盒越狱框架，采用恶意内容隐藏和内存重构以及迭代优化算法，用于自动对大型语言模型进行红队攻击。

    

    大型语言模型（LLMs）近年来取得了重要进展。在LLMs公开发布之前，人们已经做出了大量努力来将它们的行为与人类价值观保持一致。对齐的主要目标是确保它们的有益性、诚实性和无害性。然而，即使经过细致对齐的LLMs仍然容易受到恶意操纵，如越狱，导致意外的行为。越狱是有意开发恶意提示，从LLM安全限制中逃脱以生成未经审查的有害内容。以前的工作探索了不同的越狱方法来对LLMs进行红队攻击，但它们在效果和可伸缩性方面遇到了挑战。在这项工作中，我们提出了Tastle，一种新颖的黑盒越狱框架，用于自动对LLMs进行红队攻击。我们设计了恶意内容隐藏和内存重构，并结合迭代优化算法来越狱LLMs。

    arXiv:2403.08424v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved significant advances in recent days. Extensive efforts have been made before the public release of LLMs to align their behaviors with human values. The primary goal of alignment is to ensure their helpfulness, honesty and harmlessness. However, even meticulously aligned LLMs remain vulnerable to malicious manipulations such as jailbreaking, leading to unintended behaviors. The jailbreak is to intentionally develop a malicious prompt that escapes from the LLM security restrictions to produce uncensored detrimental contents. Previous works explore different jailbreak methods for red teaming LLMs, yet they encounter challenges regarding to effectiveness and scalability. In this work, we propose Tastle, a novel black-box jailbreak framework for automated red teaming of LLMs. We designed malicious content concealing and memory reframing with an iterative optimization algorithm to jailbreak LLMs, mo
    
[^2]: WaterMax: 打破LLM水印可检测性-稳健性-质量的平衡

    WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off

    [https://arxiv.org/abs/2403.04808](https://arxiv.org/abs/2403.04808)

    WaterMax提出了一种新的水印方案，能够在保持生成文本质量的同时实现高检测性能，打破了水印技术中质量和稳健性之间的传统平衡。

    

    水印是阻止大型语言模型被恶意使用的技术手段。本文提出了一种称为WaterMax的新颖水印方案，具有高检测性能，同时保持原始LLM生成文本的质量。其新设计不会对LLM进行任何修改（不调整权重、对数、温度或采样技术）。WaterMax平衡了稳健性和复杂性，与文献中的水印技术相反，从根本上引发了质量和稳健性之间的平衡。其性能在理论上得到证明并经过实验证实。在最全面的基准测试套件下，它胜过所有的最先进技术。

    arXiv:2403.04808v1 Announce Type: cross  Abstract: Watermarking is a technical means to dissuade malfeasant usage of Large Language Models. This paper proposes a novel watermarking scheme, so-called WaterMax, that enjoys high detectability while sustaining the quality of the generated text of the original LLM. Its new design leaves the LLM untouched (no modification of the weights, logits, temperature, or sampling technique). WaterMax balances robustness and complexity contrary to the watermarking techniques of the literature inherently provoking a trade-off between quality and robustness. Its performance is both theoretically proven and experimentally validated. It outperforms all the SotA techniques under the most complete benchmark suite.
    
[^3]: 欺骗的艺术：使用动态触发器的强健后门攻击

    The Art of Deception: Robust Backdoor Attack using Dynamic Stacking of Triggers. (arXiv:2401.01537v1 [cs.CR])

    [http://arxiv.org/abs/2401.01537](http://arxiv.org/abs/2401.01537)

    这项研究介绍了一种使用动态触发器进行强健后门攻击的方法，通过巧妙设计的调整，使损坏的样本与干净的样本无法区分，实验证明这种方法可以成功地欺骗语音识别系统。

    

    由于人工智能行业的最新进展，机器学习作为服务（MLaaS）领域正在经历增长的实施。然而，这种增长引发了对AI防御机制的担忧，特别是对于来自不完全可信的第三方提供商的潜在隐蔽攻击。最近的研究发现，听觉后门可能使用某些修改作为其启动机制。DynamicTrigger作为一种方法被引入，用于进行使用巧妙设计的调整来确保损坏的样本与干净的样本无法区分的动态后门攻击。通过利用波动的信号采样率，并通过动态声音触发器（比如拍手声）对说话者身份进行掩盖，可以欺骗语音识别系统（ASR）。我们的实证测试表明，DynamicTrigger在隐蔽攻击中既有效又隐蔽，并在攻击过程中取得了令人印象深刻的成功率。

    The area of Machine Learning as a Service (MLaaS) is experiencing increased implementation due to recent advancements in the AI (Artificial Intelligence) industry. However, this spike has prompted concerns regarding AI defense mechanisms, specifically regarding potential covert attacks from third-party providers that cannot be entirely trusted. Recent research has uncovered that auditory backdoors may use certain modifications as their initiating mechanism. DynamicTrigger is introduced as a methodology for carrying out dynamic backdoor attacks that use cleverly designed tweaks to ensure that corrupted samples are indistinguishable from clean. By utilizing fluctuating signal sampling rates and masking speaker identities through dynamic sound triggers (such as the clapping of hands), it is possible to deceive speech recognition systems (ASR). Our empirical testing demonstrates that DynamicTrigger is both potent and stealthy, achieving impressive success rates during covert attacks while 
    
[^4]: CyberForce: 一个用于恶意软件缓解的联邦强化学习框架

    CyberForce: A Federated Reinforcement Learning Framework for Malware Mitigation. (arXiv:2308.05978v1 [cs.CR])

    [http://arxiv.org/abs/2308.05978](http://arxiv.org/abs/2308.05978)

    CyberForce是一个联邦强化学习框架，用于在物联网设备中协同私密地确定适合缓解各种零日攻击的MTD技术。它整合了设备指纹识别和异常检测，并通过奖励或惩罚FRL agent选择的MTD机制来提高网络安全性。

    

    互联网物联网(IoT)范例的扩展是不可避免的，但是对于IoT设备对恶意软件事件的脆弱性已成为一个越来越关注的问题。最近的研究显示，将强化学习与移动目标防御(MTD)机制相结合，可以增强IoT设备的网络安全性。然而，大量的新恶意软件攻击和代理人学习和选择有效的MTD技术所需的时间使得这种方法在现实世界的IoT场景中不切实际。为解决这个问题，本研究提出了CyberForce，一个采用联邦强化学习(FRL)的框架，用于集体且保密地确定适合缓解各种零日攻击的MTD技术。CyberForce结合了设备指纹识别和异常检测，通过奖励或惩罚FRL agent选择的MTD机制。该框架在一个由十台真实IoT平台设备组成的联邦中进行了评估。通过六个恶意软件样本进行了一系列实验。

    The expansion of the Internet-of-Things (IoT) paradigm is inevitable, but vulnerabilities of IoT devices to malware incidents have become an increasing concern. Recent research has shown that the integration of Reinforcement Learning with Moving Target Defense (MTD) mechanisms can enhance cybersecurity in IoT devices. Nevertheless, the numerous new malware attacks and the time that agents take to learn and select effective MTD techniques make this approach impractical for real-world IoT scenarios. To tackle this issue, this work presents CyberForce, a framework that employs Federated Reinforcement Learning (FRL) to collectively and privately determine suitable MTD techniques for mitigating diverse zero-day attacks. CyberForce integrates device fingerprinting and anomaly detection to reward or penalize MTD mechanisms chosen by an FRL-based agent. The framework has been evaluated in a federation consisting of ten devices of a real IoT platform. A pool of experiments with six malware samp
    

