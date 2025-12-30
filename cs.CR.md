# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralised, Scalable and Privacy-Preserving Synthetic Data Generation.](http://arxiv.org/abs/2310.20062) | 这篇论文介绍了一种去中心化、可扩展且保护隐私的合成数据生成系统，使真实数据的贡献者能够参与差分隐私合成数据生成，从而提供更好的隐私和统计保证，并在机器学习流程中更好地利用合成数据。 |
| [^2] | [Prompt Injection attack against LLM-integrated Applications.](http://arxiv.org/abs/2306.05499) | 本研究分析了LLM集成应用中的提示注入攻击的复杂性和影响，提出了一种新颖的黑盒提示注入攻击技术HouYi，并揭示了应用程序提示机制中以前未知和严重低估的漏洞。我们的研究呼吁进一步开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。 |

# 详细

[^1]: 去中心化、可扩展且保护隐私的合成数据生成

    Decentralised, Scalable and Privacy-Preserving Synthetic Data Generation. (arXiv:2310.20062v1 [cs.CR])

    [http://arxiv.org/abs/2310.20062](http://arxiv.org/abs/2310.20062)

    这篇论文介绍了一种去中心化、可扩展且保护隐私的合成数据生成系统，使真实数据的贡献者能够参与差分隐私合成数据生成，从而提供更好的隐私和统计保证，并在机器学习流程中更好地利用合成数据。

    

    合成数据作为一种有潜力的方式在降低隐私风险的同时发挥数据价值。合成数据的潜力不仅局限于隐私友好的数据发布，还包括在培训机器学习算法等使用案例中补充真实数据，使其更公平、更能抵抗分布转变等。对于提供更好的隐私和统计保证以及更好地在机器学习流程中利用合成数据的算法进展引起了广泛兴趣。然而，对于负责任和值得信赖的合成数据生成来说，仅关注这些算法方面是不够的，而应该考虑合成数据生成流程的整体视角。我们构建了一个新的系统，允许真实数据的贡献者在没有依赖于值得信赖的中心的情况下自主参与差分隐私合成数据生成。我们的模块化、通用化和可扩展的解决方案基于...

    Synthetic data is emerging as a promising way to harness the value of data, while reducing privacy risks. The potential of synthetic data is not limited to privacy-friendly data release, but also includes complementing real data in use-cases such as training machine learning algorithms that are more fair and robust to distribution shifts etc. There is a lot of interest in algorithmic advances in synthetic data generation for providing better privacy and statistical guarantees and for its better utilisation in machine learning pipelines. However, for responsible and trustworthy synthetic data generation, it is not sufficient to focus only on these algorithmic aspects and instead, a holistic view of the synthetic data generation pipeline must be considered. We build a novel system that allows the contributors of real data to autonomously participate in differentially private synthetic data generation without relying on a trusted centre. Our modular, general and scalable solution is based
    
[^2]: LLM集成应用中的提示注入攻击研究

    Prompt Injection attack against LLM-integrated Applications. (arXiv:2306.05499v1 [cs.CR])

    [http://arxiv.org/abs/2306.05499](http://arxiv.org/abs/2306.05499)

    本研究分析了LLM集成应用中的提示注入攻击的复杂性和影响，提出了一种新颖的黑盒提示注入攻击技术HouYi，并揭示了应用程序提示机制中以前未知和严重低估的漏洞。我们的研究呼吁进一步开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。

    

    大语言模型(LLM)因其卓越的语言理解和生成能力而在它们周围刺激了一个充满活力的应用生态系统。然而，它们在各种服务中的广泛融合带来了重大的安全风险。本研究将解构实际LLM集成应用中的提示注入攻击的复杂性和影响。最初，我们对十个商业应用程序进行了探索性分析，突出了目前攻击策略在实践中的约束条件。受这些限制的启发，我们随后制定了HouYi，一种新颖的黑盒提示注入攻击技术，它借鉴了传统的Web注入攻击。HouYi分为三个关键元素: 一个无缝集成的预构建提示、一个注入提示诱导上下文分区以及一个恶意载荷，旨在实现攻击目标。利用HouYi，我们揭示了应用程序提示机制中以前未知和严重低估的漏洞，并演示了绕过最先进的检测机制的可行性。我们的研究呼吁进一步研究开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。

    Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and sev
    

