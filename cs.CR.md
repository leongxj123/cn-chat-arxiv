# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimistic Verifiable Training by Controlling Hardware Nondeterminism](https://arxiv.org/abs/2403.09603) | 提出了一种方法，结合了在比目标模型更高精度下进行训练、在中间计算步骤后进行四舍五入，并基于自适应阈值存储四舍五入决策，以应对硬件非确定性对训练过程的影响。 |
| [^2] | [Prompt Injection Attacks and Defenses in LLM-Integrated Applications.](http://arxiv.org/abs/2310.12815) | 本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。 |
| [^3] | [Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective.](http://arxiv.org/abs/2306.00353) | 本研究提出了一个基于概率视角的对抗样本构建方法，可以生成语义感知的对抗性样本，并可以有效规避传统对抗性攻击的强化对抗训练方法。 |

# 详细

[^1]: 控制硬件非确定性进行乐观可验证训练

    Optimistic Verifiable Training by Controlling Hardware Nondeterminism

    [https://arxiv.org/abs/2403.09603](https://arxiv.org/abs/2403.09603)

    提出了一种方法，结合了在比目标模型更高精度下进行训练、在中间计算步骤后进行四舍五入，并基于自适应阈值存储四舍五入决策，以应对硬件非确定性对训练过程的影响。

    

    AI系统日益增加的计算需求导致了为缺乏必要资源的客户进行模型训练的服务的出现。然而，确保训练的正确性并防范潜在的训练时攻击，例如数据毒化，都带来了挑战。现有的关于可验证训练的工作主要分为两类：基于证明的系统，由于需要加密技术而难以扩展，以及考虑到一个可信第三方审计员复制训练过程的“乐观”方法。 后者的一个关键挑战是，在训练期间GPU类型之间的硬件非确定性阻止审计员精确复制训练过程，因此这样的方案不够健壮。我们提出了一种方法，将训练在比目标模型更高的精度下进行，中间计算步骤后四舍五入，基于自适应阈值存储四舍五入决策。

    arXiv:2403.09603v1 Announce Type: cross  Abstract: The increasing compute demands of AI systems has led to the emergence of services that train models on behalf of clients lacking necessary resources. However, ensuring correctness of training and guarding against potential training-time attacks, such as data poisoning, poses challenges. Existing works on verifiable training largely fall into two classes: proof-based systems, which struggle to scale due to requiring cryptographic techniques, and "optimistic" methods that consider a trusted third-party auditor who replicates the training process. A key challenge with the latter is that hardware nondeterminism between GPU types during training prevents an auditor from replicating the training process exactly, and such schemes are therefore non-robust. We propose a method that combines training in a higher precision than the target model, rounding after intermediate computation steps, and storing rounding decisions based on an adaptive thr
    
[^2]: LLM-集成应用中的提示注入攻击和防御

    Prompt Injection Attacks and Defenses in LLM-Integrated Applications. (arXiv:2310.12815v1 [cs.CR])

    [http://arxiv.org/abs/2310.12815](http://arxiv.org/abs/2310.12815)

    本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。

    

    大型语言模型（LLMs）越来越多地用作各种称为LLM-集成应用的实际应用程序的后端。最近的多项研究表明，LLM-集成应用容易受到提示注入攻击的威胁，攻击者可以将恶意指令/数据注入这些应用程序的输入中，以达到攻击者的预期结果。然而，现有的研究仅限于案例研究，缺乏对提示注入攻击及其防御的系统理解。本论文旨在填补这一空白。我们提出了一个通用框架来形式化提示注入攻击，并将研究论文和博客文章中讨论的现有攻击视为我们框架的特例。我们的框架使我们能够通过组合现有攻击设计新的攻击方式。此外，我们还提出了一个系统化提示注入攻击防御的框架。利用我们的框架，我们可以预防和缓解这种类型的攻击。

    Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we con
    
[^3]: 从概率角度构建语义感知的对抗样本

    Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective. (arXiv:2306.00353v1 [stat.ML])

    [http://arxiv.org/abs/2306.00353](http://arxiv.org/abs/2306.00353)

    本研究提出了一个基于概率视角的对抗样本构建方法，可以生成语义感知的对抗性样本，并可以有效规避传统对抗性攻击的强化对抗训练方法。

    

    本研究提出了一种新颖的概率视角对抗样本构建方法——箱约束 Langevin Monte Carlo（LMC）。从这个角度出发，我们开发了一种创新性的方法，以原则性的方式生成语义感知的对抗性样本。这种方法超越了几何距离所施加的限制，选择了语义约束。我们的方法赋予了个体将其对语义的理解融入到模型中的能力。通过人类评估，我们验证了我们的语义感知的对抗样本保持其固有的含义。在 MNIST 和 SVHN 数据集上的实验结果表明，我们的语义感知的对抗样本可以有效地规避针对传统对抗性攻击的强健性对抗训练方法。

    In this study, we introduce a novel, probabilistic viewpoint on adversarial examples, achieved through box-constrained Langevin Monte Carlo (LMC). Proceeding from this perspective, we develop an innovative approach for generating semantics-aware adversarial examples in a principled manner. This methodology transcends the restriction imposed by geometric distance, instead opting for semantic constraints. Our approach empowers individuals to incorporate their personal comprehension of semantics into the model. Through human evaluation, we validate that our semantics-aware adversarial examples maintain their inherent meaning. Experimental findings on the MNIST and SVHN datasets demonstrate that our semantics-aware adversarial examples can effectively circumvent robust adversarial training methods tailored for traditional adversarial attacks.
    

