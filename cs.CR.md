# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Conflict of Robustness and Learning in Collaborative Machine Learning](https://arxiv.org/abs/2402.13700) | 在协作机器学习中，研究人员正式规范了稳健聚合器的领域，并发现现有的稳健聚合器无法实现其目标，要么无法准确识别有针对性的恶意更新，要么方法成功率不够。 |
| [^2] | [Privacy-preserving data release leveraging optimal transport and particle gradient descent](https://arxiv.org/abs/2401.17823) | 该研究提出了一种基于边际的保护隐私数据合成方法PrivPGD，利用了最优输运和粒子梯度下降的工具。该方法在不同领域的数据集上表现出色，具有高度的可扩展性和灵活性，并可以满足特定的领域约束条件。 |
| [^3] | [Finetuning Large Language Models for Vulnerability Detection.](http://arxiv.org/abs/2401.17010) | 本文优化了大规模语言模型用于源代码中的漏洞检测任务，通过微调最先进的代码语言模型WizardCoder并改进其训练过程和策略，实现了对漏洞数据集的分类性能的提升。 |
| [^4] | [FLAIM: AIM-based Synthetic Data Generation in the Federated Setting.](http://arxiv.org/abs/2310.03447) | FLAIM是一个在联邦设置中基于AIM的合成数据生成方法，该方法解决了差分隐私方向的技术在联邦场景下的适用问题，并提出了FLAIM方法来维持较高的效用和处理异构性。 |
| [^5] | [LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins.](http://arxiv.org/abs/2309.10254) | 本文提出了一个框架，用于分析和改进当前和未来与插件集成的LLM平台的安全性、隐私和安全性。在应用框架于OpenAI的插件生态系统时，我们发现了一些具体证明了潜在问题的插件。 |
| [^6] | [ChatGPT for Software Security: Exploring the Strengths and Limitations of ChatGPT in the Security Applications.](http://arxiv.org/abs/2307.12488) | 本文通过对ChatGPT在安全导向的程序分析中的表现进行研究，旨在了解其优势和局限性。研究结果可以帮助我们更好地理解ChatGPT在安全领域的应用潜力。 |

# 详细

[^1]: 在协作机器学习中稳健性和学习的冲突

    On the Conflict of Robustness and Learning in Collaborative Machine Learning

    [https://arxiv.org/abs/2402.13700](https://arxiv.org/abs/2402.13700)

    在协作机器学习中，研究人员正式规范了稳健聚合器的领域，并发现现有的稳健聚合器无法实现其目标，要么无法准确识别有针对性的恶意更新，要么方法成功率不够。

    

    协作机器学习（CML）允许参与者共同训练机器学习模型，同时保持他们的训练数据私密。在隐私是一个强烈要求的情况下，比如健康相关应用中，安全也是首要关注的问题。这意味着保护隐私的CML流程必须产生能够输出正确可靠决策的模型，甚至在可能不受信任参与者的情况下也是如此。为了解决这个问题，研究人员提出使用依赖于帮助过滤可能危及训练过程的恶意贡献的度量的“稳健聚合器”。在这项工作中，我们在文献中规范化了稳健聚合器的进展。我们的规范化能够表明现有的稳健聚合器无法实现其目标：无论是它们使用无法准确识别有针对性的恶意更新的基于距离的度量；还是提出的方法成功率不够。

    arXiv:2402.13700v1 Announce Type: new  Abstract: Collaborative Machine Learning (CML) allows participants to jointly train a machine learning model while keeping their training data private. In scenarios where privacy is a strong requirement, such as health-related applications, safety is also a primary concern. This means that privacy-preserving CML processes must produce models that output correct and reliable decisions \emph{even in the presence of potentially untrusted participants}. In response to this issue, researchers propose to use \textit{robust aggregators} that rely on metrics which help filter out malicious contributions that could compromise the training process. In this work, we formalize the landscape of robust aggregators in the literature. Our formalization allows us to show that existing robust aggregators cannot fulfill their goal: either they use distance-based metrics that cannot accurately identify targeted malicious updates; or propose methods whose success is i
    
[^2]: 采用最优输运和粒子梯度下降的保护隐私数据发布方法

    Privacy-preserving data release leveraging optimal transport and particle gradient descent

    [https://arxiv.org/abs/2401.17823](https://arxiv.org/abs/2401.17823)

    该研究提出了一种基于边际的保护隐私数据合成方法PrivPGD，利用了最优输运和粒子梯度下降的工具。该方法在不同领域的数据集上表现出色，具有高度的可扩展性和灵活性，并可以满足特定的领域约束条件。

    

    我们提出了一种新颖的方法，用于保护关键领域（如医疗保健和政府）中隐私的表格数据差分私有数据合成的任务。目前的最先进方法主要使用基于边际的方法，从私有边际估计生成数据集。在本文中，我们引入了PrivPGD，一种基于边际的私有数据合成的新一代方法，利用了最优输运和粒子梯度下降的工具。我们的算法在大范围数据集上优于现有方法，并具有高度可扩展性和灵活性，可以结合其他领域特定的约束条件。

    We present a novel approach for differentially private data synthesis of protected tabular datasets, a relevant task in highly sensitive domains such as healthcare and government. Current state-of-the-art methods predominantly use marginal-based approaches, where a dataset is generated from private estimates of the marginals. In this paper, we introduce PrivPGD, a new generation method for marginal-based private data synthesis, leveraging tools from optimal transport and particle gradient descent. Our algorithm outperforms existing methods on a large range of datasets while being highly scalable and offering the flexibility to incorporate additional domain-specific constraints.
    
[^3]: 优化大规模语言模型用于漏洞检测

    Finetuning Large Language Models for Vulnerability Detection. (arXiv:2401.17010v1 [cs.CR])

    [http://arxiv.org/abs/2401.17010](http://arxiv.org/abs/2401.17010)

    本文优化了大规模语言模型用于源代码中的漏洞检测任务，通过微调最先进的代码语言模型WizardCoder并改进其训练过程和策略，实现了对漏洞数据集的分类性能的提升。

    

    本文介绍了对大规模语言模型进行微调，并将其用于源代码中的漏洞检测的结果。我们利用最先进的语言模型StarCoder的改进版本WizardCoder，并通过进一步微调将其适应于漏洞检测任务。为了加速训练，我们修改了WizardCoder的训练过程，并探究了最佳的训练策略。针对负样本远多于正样本的不平衡数据集，我们还尝试了不同的技术来提高分类性能。微调后的WizardCoder模型在平衡和不平衡的漏洞数据集上在ROC AUC和F1度量上实现了改进，证明了将预训练的语言模型用于源代码中的漏洞检测的有效性。主要贡献包括对最先进的代码语言模型WizardCoder进行微调，提高其训练速度而不影响性能，并对训练过程和策略进行了优化。

    This paper presents the results of finetuning large language models (LLMs) for the task of detecting vulnerabilities in source code. We leverage WizardCoder, a recent improvement of the state-of-the-art LLM StarCoder, and adapt it for vulnerability detection through further finetuning. To accelerate training, we modify WizardCoder's training procedure, also we investigate optimal training regimes. For the imbalanced dataset with many more negative examples than positive, we also explore different techniques to improve classification performance. The finetuned WizardCoder model achieves improvement in ROC AUC and F1 measures on balanced and imbalanced vulnerability datasets over CodeBERT-like model, demonstrating the effectiveness of adapting pretrained LLMs for vulnerability detection in source code. The key contributions are finetuning the state-of-the-art code LLM, WizardCoder, increasing its training speed without the performance harm, optimizing the training procedure and regimes, 
    
[^4]: FLAIM: 在联邦设置中基于AIM的合成数据生成

    FLAIM: AIM-based Synthetic Data Generation in the Federated Setting. (arXiv:2310.03447v1 [cs.CR])

    [http://arxiv.org/abs/2310.03447](http://arxiv.org/abs/2310.03447)

    FLAIM是一个在联邦设置中基于AIM的合成数据生成方法，该方法解决了差分隐私方向的技术在联邦场景下的适用问题，并提出了FLAIM方法来维持较高的效用和处理异构性。

    

    保护个人隐私同时实现协同数据共享对组织至关重要。合成数据生成是一种解决方案，它产生与私有数据的统计特性相似的人工数据。虽然在差分隐私下已经设计出了许多技术，但它们主要假设数据是集中的。然而，数据往往以联邦方式分布在多个客户端上。在这项工作中，我们开始研究联邦合成表数据生成。在AIM这个先进的中心方法的基础上，我们提出了DistAIM和FLAIM。我们展示了分发AIM是简单的，扩展了基于安全多方计算的最新方法，但需要额外的开销，使其在联邦场景中不太适用。然后，我们证明了简单地联邦AIM可能导致在异构性存在的情况下效用严重下降。为了解决这两个问题，我们提出了一种增强的FLAIM方法，该方法可以维持较高的效用，并且可以处理联邦设置中的异构性。

    Preserving individual privacy while enabling collaborative data sharing is crucial for organizations. Synthetic data generation is one solution, producing artificial data that mirrors the statistical properties of private data. While numerous techniques have been devised under differential privacy, they predominantly assume data is centralized. However, data is often distributed across multiple clients in a federated manner. In this work, we initiate the study of federated synthetic tabular data generation. Building upon a SOTA central method known as AIM, we present DistAIM and FLAIM. We show it is straightforward to distribute AIM, extending a recent approach based on secure multi-party computation which necessitates additional overhead, making it less suited to federated scenarios. We then demonstrate that naively federating AIM can lead to substantial degradation in utility under the presence of heterogeneity. To mitigate both issues, we propose an augmented FLAIM approach that mai
    
[^5]: LLM平台安全：将系统评估框架应用于OpenAI的ChatGPT插件

    LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins. (arXiv:2309.10254v1 [cs.CR])

    [http://arxiv.org/abs/2309.10254](http://arxiv.org/abs/2309.10254)

    本文提出了一个框架，用于分析和改进当前和未来与插件集成的LLM平台的安全性、隐私和安全性。在应用框架于OpenAI的插件生态系统时，我们发现了一些具体证明了潜在问题的插件。

    

    近期，如ChatGPT等大型语言模型（LLM）平台开始提供插件生态系统，以与互联网上的第三方服务进行交互。虽然这些插件扩展了LLM平台的功能，但它们是由任意的第三方开发的，因此不能隐式信任。插件还使用自然语言与LLM平台和用户进行交互，这可能导致模糊的解释。本文提出了一个框架，为LLM平台设计者分析和改进当前和未来与插件集成的LLM平台的安全性、隐私和安全性奠定了基础。我们的框架是一个攻击分类法的表述，通过迭代地探索LLM平台相关方如何利用他们的能力和责任对彼此进行攻击来开发的。作为我们迭代过程的一部分，我们将我们的框架应用于OpenAI的插件生态系统。我们揭示了一些具体证明了潜在问题的插件。

    Large language model (LLM) platforms, such as ChatGPT, have recently begun offering a plugin ecosystem to interface with third-party services on the internet. While these plugins extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Plugins also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future plugin-integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin ecosystem. We uncover plugins that concretely demonstrate the poten
    
[^6]: ChatGPT用于软件安全：探索ChatGPT在安全应用中的优点和局限性

    ChatGPT for Software Security: Exploring the Strengths and Limitations of ChatGPT in the Security Applications. (arXiv:2307.12488v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2307.12488](http://arxiv.org/abs/2307.12488)

    本文通过对ChatGPT在安全导向的程序分析中的表现进行研究，旨在了解其优势和局限性。研究结果可以帮助我们更好地理解ChatGPT在安全领域的应用潜力。

    

    作为一个多才多艺的大型语言模型，ChatGPT在各个领域应对问题的潜力得到了显著的展示。它能够分析、理解和综合来自在线资源和用户输入的信息，引起了广泛的关注。先前的研究已经探索了ChatGPT在代码生成和代码审查方面的能力。在本文中，我们深入研究了ChatGPT在面向安全的程序分析中的能力，从攻击者和安全分析师的角度进行了探讨。我们通过一个案例研究来评估ChatGPT在几个安全导向的程序分析任务中的回答质量，并有意地引入挑战来评估其响应能力。通过对ChatGPT提供的答案质量的考察，我们对其在安全导向的程序分析领域的优点和局限性有了更清晰的认识。

    ChatGPT, as a versatile large language model, has demonstrated remarkable potential in addressing inquiries across various domains. Its ability to analyze, comprehend, and synthesize information from both online sources and user inputs has garnered significant attention. Previous research has explored ChatGPT's competence in code generation and code reviews. In this paper, we delve into ChatGPT's capabilities in security-oriented program analysis, focusing on perspectives from both attackers and security analysts. We present a case study involving several security-oriented program analysis tasks while deliberately introducing challenges to assess ChatGPT's responses. Through an examination of the quality of answers provided by ChatGPT, we gain a clearer understanding of its strengths and limitations in the realm of security-oriented program analysis.
    

