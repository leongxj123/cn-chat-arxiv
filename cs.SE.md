# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics.](http://arxiv.org/abs/2401.09622) | SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。 |
| [^2] | [Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers.](http://arxiv.org/abs/2401.06461) | 本文通过分析代码的属性，揭示了机器和人类代码之间的独特模式，尤其是结构分割对于识别代码来源很关键。基于这些发现，我们提出了一种名为DetectCodeGPT的新方法来检测机器生成的代码。 |
| [^3] | [Semi-supervised learning via DQN for log anomaly detection.](http://arxiv.org/abs/2401.03151) | 本文提出了一种半监督的日志异常检测方法，命名为DQNLog，通过结合深度强化学习中的DQN算法，利用少量有标记的数据和大规模无标记的数据集，有效解决了数据不平衡和标记数量有限的问题，并且通过与异常环境交互和主动探索无标记的数据集，学习已知的异常并发现未知的异常。 |
| [^4] | [SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents.](http://arxiv.org/abs/2308.02594) | 本文提出了一种基于机器学习的安全监测方法SMARLA，用于深度强化学习智能体。该方法设计为黑盒子，利用状态抽象减少状态空间，实现对智能体状态的安全违规预测。经验证，SMARLA具有准确的违规预测能力，并可在智能体执行的早期阶段进行预测。 |

# 详细

[^1]: SMOOTHIE: 软件分析的超参数优化理论

    SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics. (arXiv:2401.09622v1 [cs.SE])

    [http://arxiv.org/abs/2401.09622](http://arxiv.org/abs/2401.09622)

    SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。

    

    超参数优化是调整学习器控制参数的黑魔法。在软件分析中，经常发现调优可以带来显著的性能改进。尽管如此，超参数优化在软件分析中通常被很少或很差地应用，可能是因为探索所有参数选项的CPU成本太高。我们假设当损失函数的“光滑度”更好时，学习器的泛化能力更强。这个理论非常有用，因为可以很快测试不同超参数选择对“光滑度”的影响（例如，对于深度学习器，在一个epoch之后就可以进行测试）。为了测试这个理论，本文实现和测试了SMOOTHIE，一种通过考虑“光滑度”来引导优化的新型超参数优化器。本文的实验将SMOOTHIE应用于多个软件工程任务，包括（a）GitHub问题寿命预测；（b）静态代码警告中错误警报的检测；（c）缺陷预测。

    Hyper-parameter optimization is the black art of tuning a learner's control parameters. In software analytics, a repeated result is that such tuning can result in dramatic performance improvements. Despite this, hyper-parameter optimization is often applied rarely or poorly in software analytics--perhaps due to the CPU cost of exploring all those parameter options can be prohibitive.  We theorize that learners generalize better when the loss landscape is ``smooth''. This theory is useful since the influence on ``smoothness'' of different hyper-parameter choices can be tested very quickly (e.g. for a deep learner, after just one epoch).  To test this theory, this paper implements and tests SMOOTHIE, a novel hyper-parameter optimizer that guides its optimizations via considerations of ``smothness''. The experiments of this paper test SMOOTHIE on numerous SE tasks including (a) GitHub issue lifetime prediction; (b) detecting false alarms in static code warnings; (c) defect prediction, and
    
[^2]: 代码之间的界限：揭示机器和人类程序员之间不同的模式

    Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers. (arXiv:2401.06461v1 [cs.SE])

    [http://arxiv.org/abs/2401.06461](http://arxiv.org/abs/2401.06461)

    本文通过分析代码的属性，揭示了机器和人类代码之间的独特模式，尤其是结构分割对于识别代码来源很关键。基于这些发现，我们提出了一种名为DetectCodeGPT的新方法来检测机器生成的代码。

    

    大型语言模型在代码生成方面取得了显著的进展，但它们模糊了机器和人类源代码之间的区别，导致软件产物的完整性和真实性问题。本文通过对代码长度、词汇多样性和自然性等属性的严格分析，揭示了机器和人类代码固有的独特模式。在我们的研究中特别注意到，代码的结构分割是识别其来源的关键因素。基于我们的发现，我们提出了一种名为DetectCodeGPT的新型机器生成代码检测方法，该方法改进了DetectGPT。

    Large language models have catalyzed an unprecedented wave in code generation. While achieving significant advances, they blur the distinctions between machine-and human-authored source code, causing integrity and authenticity issues of software artifacts. Previous methods such as DetectGPT have proven effective in discerning machine-generated texts, but they do not identify and harness the unique patterns of machine-generated code. Thus, its applicability falters when applied to code. In this paper, we carefully study the specific patterns that characterize machine and human-authored code. Through a rigorous analysis of code attributes such as length, lexical diversity, and naturalness, we expose unique pat-terns inherent to each source. We particularly notice that the structural segmentation of code is a critical factor in identifying its provenance. Based on our findings, we propose a novel machine-generated code detection method called DetectCodeGPT, which improves DetectGPT by cap
    
[^3]: 通过DQN进行半监督学习用于日志异常检测

    Semi-supervised learning via DQN for log anomaly detection. (arXiv:2401.03151v1 [cs.SE])

    [http://arxiv.org/abs/2401.03151](http://arxiv.org/abs/2401.03151)

    本文提出了一种半监督的日志异常检测方法，命名为DQNLog，通过结合深度强化学习中的DQN算法，利用少量有标记的数据和大规模无标记的数据集，有效解决了数据不平衡和标记数量有限的问题，并且通过与异常环境交互和主动探索无标记的数据集，学习已知的异常并发现未知的异常。

    

    日志异常检测在保障现代软件系统的安全性和维护方面起着关键作用。目前，检测日志数据中的异常的主要方法是通过监督式异常检测。然而，现有的监督式方法往往依赖于有标记的数据，在实际情况下往往受到限制。本文提出了一种半监督的日志异常检测方法，结合深度强化学习中的DQN算法，称为DQNLog。DQNLog利用少量有标记的数据和大规模无标记的数据集，有效解决了数据不平衡和标记数量有限的问题。该方法不仅通过与偏向于异常的环境进行交互来学习已知的异常，还通过主动探索无标记的数据集来发现未知的异常。此外，DQNLog还引入了交叉熵损失项，防止在深度强化学习中出现模型过高估计的情况。

    Log anomaly detection plays a critical role in ensuring the security and maintenance of modern software systems. At present, the primary approach for detecting anomalies in log data is through supervised anomaly detection. Nonetheless, existing supervised methods heavily rely on labeled data, which can be frequently limited in real-world scenarios. In this paper, we propose a semi-supervised log anomaly detection method that combines the DQN algorithm from deep reinforcement learning, which is called DQNLog. DQNLog leverages a small amount of labeled data and a large-scale unlabeled dataset, effectively addressing the challenges of imbalanced data and limited labeling. This approach not only learns known anomalies by interacting with an environment biased towards anomalies but also discovers unknown anomalies by actively exploring the unlabeled dataset. Additionally, DQNLog incorporates a cross-entropy loss term to prevent model overestimation during Deep Reinforcement Learning (DRL). 
    
[^4]: SMARLA：一种用于深度强化学习智能体的安全监测方法

    SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents. (arXiv:2308.02594v1 [cs.LG])

    [http://arxiv.org/abs/2308.02594](http://arxiv.org/abs/2308.02594)

    本文提出了一种基于机器学习的安全监测方法SMARLA，用于深度强化学习智能体。该方法设计为黑盒子，利用状态抽象减少状态空间，实现对智能体状态的安全违规预测。经验证，SMARLA具有准确的违规预测能力，并可在智能体执行的早期阶段进行预测。

    

    深度强化学习算法(DRL)越来越多地应用于安全关键系统。确保DRL智能体的安全性在这种情况下是一个关键问题。然而，仅依靠测试是不足以确保安全性的，因为它不能提供保证。构建安全监测器是缓解这一挑战的一种解决方案。本文提出了SMARLA，一种基于机器学习的安全监测方法，专为DRL智能体设计。出于实际原因，SMARLA被设计为黑盒子(因为它不需要访问智能体的内部)，并利用状态抽象来减少状态空间，从而促进从智能体的状态学习安全违规预测模型。我们在两个知名的RL案例研究中验证了SMARLA。经验分析表明，SMARLA具有准确的违规预测能力，误报率低，并且可以在智能体执行的一半左右的早期阶段预测安全违规。

    Deep reinforcement learning algorithms (DRL) are increasingly being used in safety-critical systems. Ensuring the safety of DRL agents is a critical concern in such contexts. However, relying solely on testing is not sufficient to ensure safety as it does not offer guarantees. Building safety monitors is one solution to alleviate this challenge. This paper proposes SMARLA, a machine learning-based safety monitoring approach designed for DRL agents. For practical reasons, SMARLA is designed to be black-box (as it does not require access to the internals of the agent) and leverages state abstraction to reduce the state space and thus facilitate the learning of safety violation prediction models from agent's states. We validated SMARLA on two well-known RL case studies. Empirical analysis reveals that SMARLA achieves accurate violation prediction with a low false positive rate, and can predict safety violations at an early stage, approximately halfway through the agent's execution before 
    

