# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness](https://rss.arxiv.org/abs/2401.15963) | 本论文提出了一个新的基准测试 NoFunEval，用于评估代码语言模型在非功能性要求和简单分类实例方面的表现。研究发现，目前的代码语言模型在处理这些要求时存在根本性的盲点。 |
| [^2] | [Trained Without My Consent: Detecting Code Inclusion In Language Models Trained on Code](https://arxiv.org/abs/2402.09299) | 这项研究关注如何在训练代码的语言模型中检测代码包含，以解决使用这些模型进行代码审计时的版权侵权问题。 |
| [^3] | [An exploratory study on automatic identification of assumptions in the development of deep learning frameworks.](http://arxiv.org/abs/2401.03653) | 本研究以构建一个新的最大假设数据集为基础，针对深度学习框架开发中手动识别假设的问题进行了探索性研究。在该研究中，我们发现手动识别假设的成本高，因此探讨了使用传统机器学习模型和流行的深度学习模型来识别假设的性能。 |
| [^4] | [RLocator: Reinforcement Learning for Bug Localization.](http://arxiv.org/abs/2305.05586) | 本文提出了一种基于强化学习的Bug定位方法RLocator，相较于其他最先进的Bug定位技术具有更优越的性能。 |

# 详细

[^1]: NoFunEval: 有趣的是，代码语言模型在超出功能正确性的要求上遇到困难

    NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness

    [https://rss.arxiv.org/abs/2401.15963](https://rss.arxiv.org/abs/2401.15963)

    本论文提出了一个新的基准测试 NoFunEval，用于评估代码语言模型在非功能性要求和简单分类实例方面的表现。研究发现，目前的代码语言模型在处理这些要求时存在根本性的盲点。

    

    现有的代码语言模型（code LMs）的评估基准几乎完全集中在LMs是否能够生成功能正确的代码上。在实际的软件工程中，开发人员会考虑超出功能正确性的要求。他们对于“如何”实现功能有着对整体系统设计目标（如效率、安全性和可维护性）的要求。如果LMs能够展示对要求和代码语义的强大理解能力，他们也会更加信任这些LMs。我们提出了一个新的基准测试NoFunEval来评估代码LMs在非功能性要求和简单分类实例方面的表现。我们提出了一个提示方法Coding Concepts (CoCo)，可以用于开发人员向LMs传达领域知识。我们对22个代码LMs进行了广泛评估，发现它们在我们的基准测试中普遍表现不佳，暗示着它们在处理这些问题时存在根本性的盲点。

    Existing evaluation benchmarks of language models of code (code LMs) focus almost exclusively on whether the LMs can generate functionally-correct code. In real-world software engineering, developers think beyond functional correctness. They have requirements on "how" a functionality should be implemented to meet overall system design objectives like efficiency, security, and maintainability. They would also trust the code LMs more if the LMs demonstrate robust understanding of requirements and code semantics.   We propose a new benchmark NoFunEval to evaluate code LMs on non-functional requirements and simple classification instances for both functional and non-functional requirements. We propose a prompting method, Coding Concepts (CoCo), as a way for a developer to communicate the domain knowledge to the LMs. We conduct an extensive evaluation of twenty-two code LMs. Our finding is that they generally falter when tested on our benchmark, hinting at fundamental blindspots in their tr
    
[^2]: 未经本人同意的训练：在训练代码的语言模型中检测代码包含

    Trained Without My Consent: Detecting Code Inclusion In Language Models Trained on Code

    [https://arxiv.org/abs/2402.09299](https://arxiv.org/abs/2402.09299)

    这项研究关注如何在训练代码的语言模型中检测代码包含，以解决使用这些模型进行代码审计时的版权侵权问题。

    

    代码审计通过验证开发的代码是否符合标准、法规和版权保护，确保其不包含来自受保护来源的代码。在软件开发过程中，大型语言模型(LLMs)作为编码助手的出现给代码审计带来了新的挑战。训练这些模型的数据集主要来自公开可用的来源。这引发了知识产权侵权问题，因为开发者的代码已包含在数据集中。因此，使用LLMs开发的代码审计具有挑战性，因为我们无法准确确定开发过程中使用的LLM是否已经在特定的受版权保护的代码上进行了训练，因为我们无法获得这些模型的训练数据集。鉴于训练数据集的保密性，传统的代码克隆检测等方法无法确保版权侵权。

    arXiv:2402.09299v1 Announce Type: cross Abstract: Code auditing ensures that the developed code adheres to standards, regulations, and copyright protection by verifying that it does not contain code from protected sources. The recent advent of Large Language Models (LLMs) as coding assistants in the software development process poses new challenges for code auditing. The dataset for training these models is mainly collected from publicly available sources. This raises the issue of intellectual property infringement as developers' codes are already included in the dataset. Therefore, auditing code developed using LLMs is challenging, as it is difficult to reliably assert if an LLM used during development has been trained on specific copyrighted codes, given that we do not have access to the training datasets of these models. Given the non-disclosure of the training datasets, traditional approaches such as code clone detection are insufficient for asserting copyright infringement. To add
    
[^3]: 关于深度学习框架开发中自动识别假设的探索性研究

    An exploratory study on automatic identification of assumptions in the development of deep learning frameworks. (arXiv:2401.03653v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2401.03653](http://arxiv.org/abs/2401.03653)

    本研究以构建一个新的最大假设数据集为基础，针对深度学习框架开发中手动识别假设的问题进行了探索性研究。在该研究中，我们发现手动识别假设的成本高，因此探讨了使用传统机器学习模型和流行的深度学习模型来识别假设的性能。

    

    利益相关方在深度学习框架开发中经常做出假设。这些假设涉及各种软件构件（例如需求、设计决策和技术债务），可能会被证明无效，从而导致系统故障。现有的假设管理方法和工具通常依赖于手动识别假设。然而，假设分散在深度学习框架开发的各种源头（例如代码注释、提交、拉取请求和问题）中，手动识别假设成本较高（例如时间和资源消耗）。为了解决深度学习框架开发中手动识别假设的问题，我们构建了一个新的并且最大的假设数据集（称为AssuEval），该数据集收集自GitHub上的TensorFlow和Keras代码库；我们探讨了七个传统的机器学习模型（例如支持向量机、分类回归树）和一个流行的深度学习模型的性能。

    Stakeholders constantly make assumptions in the development of deep learning (DL) frameworks. These assumptions are related to various types of software artifacts (e.g., requirements, design decisions, and technical debt) and can turn out to be invalid, leading to system failures. Existing approaches and tools for assumption management usually depend on manual identification of assumptions. However, assumptions are scattered in various sources (e.g., code comments, commits, pull requests, and issues) of DL framework development, and manually identifying assumptions has high costs (e.g., time and resources). To overcome the issues of manually identifying assumptions in DL framework development, we constructed a new and largest dataset (i.e., AssuEval) of assumptions collected from the TensorFlow and Keras repositories on GitHub; explored the performance of seven traditional machine learning models (e.g., Support Vector Machine, Classification and Regression Trees), a popular DL model (i
    
[^4]: RLocator: 利用强化学习进行Bug定位

    RLocator: Reinforcement Learning for Bug Localization. (arXiv:2305.05586v1 [cs.SE])

    [http://arxiv.org/abs/2305.05586](http://arxiv.org/abs/2305.05586)

    本文提出了一种基于强化学习的Bug定位方法RLocator，相较于其他最先进的Bug定位技术具有更优越的性能。

    

    软件开发者在他们的项目中花费了大量的时间来修复Bugs。为了简化这个过程，提出了Bug定位方法来确定哪些源代码文件可能是负责特定Bug的源头。之前的工作提出了几种基于相似性的机器学习技术，用于Bug定位。尽管这些技术取得了显著进展，但它们并没有直接优化评估指标。相反，在训练和测试阶段使用了不同的度量标准，这会对检索任务的模型性能产生负面影响。在本文中，我们提出了一种基于强化学习的Bug定位方法RLocator。我们使用马尔可夫决策过程（MDP）来优化评估指标，从而对Bug定位问题进行公式化。我们提出了该技术，并基于六种高度流行的Apache项目的8,316个Bug报告的基准数据集进行了实验评估。我们的评估表明，RLocator相较于其他最先进的Bug定位技术具有更优越的性能。

    Software developers spend a significant portion of time fixing bugs in their projects. To streamline this process, bug localization approaches have been proposed to identify the source code files that are likely responsible for a particular bug. Prior work proposed several similarity-based machine-learning techniques for bug localization. Despite significant advances in these techniques, they do not directly optimize the evaluation measures. Instead, they use different metrics in the training and testing phases, which can negatively impact the model performance in retrieval tasks. In this paper, we propose RLocator, a Reinforcement Learning-based (RL) bug localization approach. We formulate the bug localization problem using a Markov Decision Process (MDP) to optimize the evaluation measures directly. We present the technique and experimentally evaluate it based on a benchmark dataset of 8,316 bug reports from six highly popular Apache projects. Our evaluation shows that RLocator achie
    

