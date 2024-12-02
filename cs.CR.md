# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-shot sampling of adversarial entities in biomedical question answering](https://arxiv.org/abs/2402.10527) | 在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。 |
| [^2] | [DeepInception: Hypnotize Large Language Model to Be Jailbreaker](https://arxiv.org/abs/2311.03191) | 本研究提出了一种名为DeepInception的轻量级方法，利用语言模型的角色扮演能力构建新颖的嵌套场景，成功催眠大型语言模型成为破解者。通过实验证明，DeepInception在破解成功率方面具有竞争力，并揭示了开源和闭源语言模型的关键弱点。 |
| [^3] | [Multi-Trigger Backdoor Attacks: More Triggers, More Threats.](http://arxiv.org/abs/2401.15295) | 本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。 |
| [^4] | [Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning.](http://arxiv.org/abs/2308.04964) | 这篇论文介绍了Adversarial ModSecurity，它是一个使用强大的机器学习来对抗SQL注入攻击的防火墙。通过将核心规则集作为输入特征，该模型可以识别并防御对抗性SQL注入攻击。实验结果表明，AdvModSec在训练后能够有效地应对这类攻击。 |

# 详细

[^1]: 生物医学问题回答中的零样本采样对抗实体

    Zero-shot sampling of adversarial entities in biomedical question answering

    [https://arxiv.org/abs/2402.10527](https://arxiv.org/abs/2402.10527)

    在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。

    

    大型语言模型（LLM）中参数域知识的增加深度推动它们在现实世界应用中的快速部署。在高风险和知识密集型任务中，理解模型的漏洞对于量化模型预测的可信度和规范其使用至关重要。最近发现在自然语言处理任务中作为对抗示例的命名实体引发了关于它们在其他环境中可能的伪装的疑问。在这里，我们提出了一种在嵌入空间中的幂缩放距离加权采样方案，以发现多样化的对抗实体作为干扰因素。我们展示了它在生物医学主题的对抗性问题回答中优于随机采样的优势。我们的方法使得可以探索攻击表面上的不同区域，这揭示了两种在特征上明显不同的对抗性实体的制度。此外，我们展示了攻击方式如何...

    arXiv:2402.10527v1 Announce Type: new  Abstract: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks su
    
[^2]: DeepInception: 催眠大型语言模型成为破解者

    DeepInception: Hypnotize Large Language Model to Be Jailbreaker

    [https://arxiv.org/abs/2311.03191](https://arxiv.org/abs/2311.03191)

    本研究提出了一种名为DeepInception的轻量级方法，利用语言模型的角色扮演能力构建新颖的嵌套场景，成功催眠大型语言模型成为破解者。通过实验证明，DeepInception在破解成功率方面具有竞争力，并揭示了开源和闭源语言模型的关键弱点。

    

    尽管大型语言模型（LLMs）在各种应用中取得了显著的成功，但它们容易受到破解攻击，使得安全措施无效。然而，以往的破解研究通常采用暴力优化或高计算成本的外推方法，这可能并不实际或有效。本文受到以米尔格拉姆实验为灵感，关于权威力量对于引发有害行为的影响，我们提出了一种轻量级的方法，称为DeepInception，可以轻松地催眠LLM成为破解者。具体而言，DeepInception利用LLM的角色扮演能力构建了一个新颖的嵌套场景来行为，实现了在正常场景下逃避使用控制的自适应方式。实验结果表明，我们的DeepInception在破解成功率方面与以往的方法竞争力相当，并可以在后续交互中实现持续的破解，揭示了开源和闭源LLM的自失关键弱点。

    Despite remarkable success in various applications, large language models (LLMs) are vulnerable to adversarial jailbreaks that make the safety guardrails void. However, previous studies for jailbreaks usually resort to brute-force optimization or extrapolations of a high computation cost, which might not be practical or effective. In this paper, inspired by the Milgram experiment w.r.t. the authority power for inciting harmfulness, we disclose a lightweight method, termed DeepInception, which can easily hypnotize LLM to be a jailbreaker. Specifically, DeepInception leverages the personification ability of LLM to construct a novel nested scene to behave, which realizes an adaptive way to escape the usage control in a normal scenario. Empirically, our DeepInception can achieve competitive jailbreak success rates with previous counterparts and realize a continuous jailbreak in subsequent interactions, which reveals the critical weakness of self-losing on both open and closed-source LLMs l
    
[^3]: 多触发后门攻击：更多触发器，更多威胁

    Multi-Trigger Backdoor Attacks: More Triggers, More Threats. (arXiv:2401.15295v1 [cs.LG])

    [http://arxiv.org/abs/2401.15295](http://arxiv.org/abs/2401.15295)

    本文主要研究了多触发后门攻击对深度神经网络的威胁。通过提出并研究了三种类型的多触发攻击，包括并行、顺序和混合攻击，文章揭示了不同触发器对同一数据集的共存、覆写和交叉激活效果。结果表明单触发攻击容易引起覆写问题。

    

    后门攻击已经成为深度神经网络（DNNs）的（预）训练和部署的主要威胁。尽管后门攻击在一些研究中已经得到了广泛的探讨，但其中大部分都集中在使用单个类型的触发器来污染数据集的单触发攻击上。可以说，在现实世界中，后门攻击可能更加复杂，例如，同一数据集可能存在多个对手，如果该数据集具有较高的价值。在这项工作中，我们研究了在多触发攻击设置下后门攻击的实际威胁，多个对手利用不同类型的触发器来污染同一数据集。通过提出和研究并行、顺序和混合攻击这三种类型的多触发攻击，我们提供了关于不同触发器对同一数据集的共存、覆写和交叉激活效果的重要认识。此外，我们还展示了单触发攻击往往容易引起覆写问题。

    Backdoor attacks have emerged as a primary threat to (pre-)training and deployment of deep neural networks (DNNs). While backdoor attacks have been extensively studied in a body of works, most of them were focused on single-trigger attacks that poison a dataset using a single type of trigger. Arguably, real-world backdoor attacks can be much more complex, e.g., the existence of multiple adversaries for the same dataset if it is of high value. In this work, we investigate the practical threat of backdoor attacks under the setting of \textbf{multi-trigger attacks} where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks, including parallel, sequential, and hybrid attacks, we provide a set of important understandings of the coexisting, overwriting, and cross-activating effects between different triggers on the same dataset. Moreover, we show that single-trigger attacks tend to cause over
    
[^4]: Adversarial ModSecurity: 使用强大的机器学习对抗SQL注入攻击

    Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning. (arXiv:2308.04964v1 [cs.LG])

    [http://arxiv.org/abs/2308.04964](http://arxiv.org/abs/2308.04964)

    这篇论文介绍了Adversarial ModSecurity，它是一个使用强大的机器学习来对抗SQL注入攻击的防火墙。通过将核心规则集作为输入特征，该模型可以识别并防御对抗性SQL注入攻击。实验结果表明，AdvModSec在训练后能够有效地应对这类攻击。

    

    ModSecurity被广泛认可为标准的开源Web应用防火墙(WAF)，由OWASP基金会维护。它通过与核心规则集进行匹配来检测恶意请求，识别出常见的攻击模式。每个规则在CRS中都被手动分配一个权重，基于相应攻击的严重程度，如果触发规则的权重之和超过给定的阈值，就会被检测为恶意请求。然而，我们的研究表明，这种简单的策略在检测SQL注入攻击方面很不有效，因为它往往会阻止许多合法请求，同时还容易受到对抗性SQL注入攻击的影响，即故意操纵以逃避检测的攻击。为了克服这些问题，我们设计了一个名为AdvModSec的强大机器学习模型，它将CRS规则作为输入特征，并经过训练以检测对抗性SQL注入攻击。我们的实验表明，AdvModSec在针对该攻击的流量上进行训练后表现出色。

    ModSecurity is widely recognized as the standard open-source Web Application Firewall (WAF), maintained by the OWASP Foundation. It detects malicious requests by matching them against the Core Rule Set, identifying well-known attack patterns. Each rule in the CRS is manually assigned a weight, based on the severity of the corresponding attack, and a request is detected as malicious if the sum of the weights of the firing rules exceeds a given threshold. In this work, we show that this simple strategy is largely ineffective for detecting SQL injection (SQLi) attacks, as it tends to block many legitimate requests, while also being vulnerable to adversarial SQLi attacks, i.e., attacks intentionally manipulated to evade detection. To overcome these issues, we design a robust machine learning model, named AdvModSec, which uses the CRS rules as input features, and it is trained to detect adversarial SQLi attacks. Our experiments show that AdvModSec, being trained on the traffic directed towa
    

