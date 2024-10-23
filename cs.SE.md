# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ansible Lightspeed: A Code Generation Service for IT Automation](https://arxiv.org/abs/2402.17442) | Ansible Lightspeed是一种基于大型语言模型的服务，专注于将自然语言转换为Ansible代码，为IT自动化领域带来了创新。 |
| [^2] | [Towards Enhancing the Reproducibility of Deep Learning Bugs: An Empirical Study.](http://arxiv.org/abs/2401.03069) | 本研究旨在提高深度学习Bug的可复现性，通过构建数据集和确定编辑动作和有用信息，这能够解决目前研究中忽视的问题。 |
| [^3] | [SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents.](http://arxiv.org/abs/2308.02594) | 本文提出了一种基于机器学习的安全监测方法SMARLA，用于深度强化学习智能体。该方法设计为黑盒子，利用状态抽象减少状态空间，实现对智能体状态的安全违规预测。经验证，SMARLA具有准确的违规预测能力，并可在智能体执行的早期阶段进行预测。 |

# 详细

[^1]: Ansible Lightspeed: 一种用于IT自动化的代码生成服务

    Ansible Lightspeed: A Code Generation Service for IT Automation

    [https://arxiv.org/abs/2402.17442](https://arxiv.org/abs/2402.17442)

    Ansible Lightspeed是一种基于大型语言模型的服务，专注于将自然语言转换为Ansible代码，为IT自动化领域带来了创新。

    

    大型语言模型（LLMs）的问世使得创建可提高开发者生产力的工具成为可能，集成开发环境（IDEs）常被用作与LLMs交互的接口。已发布许多这类工具，但几乎全部都专注于通用编程语言，很少关注对IT自动化至关重要的特定领域语言。Ansible是一种基于YAML的IT自动化特定语言。Red Hat Ansible Lightspeed与IBM Watson Code Assistant合作的Ansible Lightspeed是一种基于LLM的服务，专门用于将自然语言转换为Ansible代码。

    arXiv:2402.17442v1 Announce Type: cross  Abstract: The availability of Large Language Models (LLMs) which can generate code, has made it possible to create tools that improve developer productivity. Integrated development environments or IDEs which developers use to write software are often used as an interface to interact with LLMs. Although many such tools have been released, almost all of them focus on general-purpose programming languages. Domain-specific languages, such as those crucial for IT automation, have not received much attention. Ansible is one such YAML-based IT automation-specific language. Red Hat Ansible Lightspeed with IBM Watson Code Assistant, further referred to as Ansible Lightspeed, is an LLM-based service designed explicitly for natural language to Ansible code generation.   In this paper, we describe the design and implementation of the Ansible Lightspeed service and analyze feedback from thousands of real users. We examine diverse performance indicators, clas
    
[^2]: 提高深度学习Bug可复现性的探索性研究

    Towards Enhancing the Reproducibility of Deep Learning Bugs: An Empirical Study. (arXiv:2401.03069v1 [cs.SE])

    [http://arxiv.org/abs/2401.03069](http://arxiv.org/abs/2401.03069)

    本研究旨在提高深度学习Bug的可复现性，通过构建数据集和确定编辑动作和有用信息，这能够解决目前研究中忽视的问题。

    

    背景：深度学习在各个领域取得了显著进展。然而，与传统软件系统一样，深度学习系统也存在Bug，这可能对自动驾驶等领域产生严重影响。尽管深度学习技术取得了重大进展，但很少有研究关注深度学习Bug的可复现性，这阻碍了Bug的解决。现有文献指出，仅有3%的深度学习Bug是可复现的，这凸显了进一步研究的必要性。目标：本文考察深度学习Bug的可复现性，识别可提高深度学习Bug可复现性的编辑动作和有用信息。方法：首先，构建了一个包含来自Stack Overflow和Defects4ML的3个框架和22个架构的668个深度学习Bug的数据集。其次，使用分层抽样选择了102个Bug，并尝试确定它们的可复现性。在复现这些Bug的过程中，我们识别了编辑动作和有用信息。

    Context: Deep learning has achieved remarkable progress in various domains. However, like traditional software systems, deep learning systems contain bugs, which can have severe impacts, as evidenced by crashes involving autonomous vehicles. Despite substantial advancements in deep learning techniques, little research has focused on reproducing deep learning bugs, which hinders resolving them. Existing literature suggests that only 3% of deep learning bugs are reproducible, underscoring the need for further research.  Objective: This paper examines the reproducibility of deep learning bugs. We identify edit actions and useful information that could improve deep learning bug reproducibility.  Method: First, we construct a dataset of 668 deep learning bugs from Stack Overflow and Defects4ML across 3 frameworks and 22 architectures. Second, we select 102 bugs using stratified sampling and try to determine their reproducibility. While reproducing these bugs, we identify edit actions and us
    
[^3]: SMARLA：一种用于深度强化学习智能体的安全监测方法

    SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents. (arXiv:2308.02594v1 [cs.LG])

    [http://arxiv.org/abs/2308.02594](http://arxiv.org/abs/2308.02594)

    本文提出了一种基于机器学习的安全监测方法SMARLA，用于深度强化学习智能体。该方法设计为黑盒子，利用状态抽象减少状态空间，实现对智能体状态的安全违规预测。经验证，SMARLA具有准确的违规预测能力，并可在智能体执行的早期阶段进行预测。

    

    深度强化学习算法(DRL)越来越多地应用于安全关键系统。确保DRL智能体的安全性在这种情况下是一个关键问题。然而，仅依靠测试是不足以确保安全性的，因为它不能提供保证。构建安全监测器是缓解这一挑战的一种解决方案。本文提出了SMARLA，一种基于机器学习的安全监测方法，专为DRL智能体设计。出于实际原因，SMARLA被设计为黑盒子(因为它不需要访问智能体的内部)，并利用状态抽象来减少状态空间，从而促进从智能体的状态学习安全违规预测模型。我们在两个知名的RL案例研究中验证了SMARLA。经验分析表明，SMARLA具有准确的违规预测能力，误报率低，并且可以在智能体执行的一半左右的早期阶段预测安全违规。

    Deep reinforcement learning algorithms (DRL) are increasingly being used in safety-critical systems. Ensuring the safety of DRL agents is a critical concern in such contexts. However, relying solely on testing is not sufficient to ensure safety as it does not offer guarantees. Building safety monitors is one solution to alleviate this challenge. This paper proposes SMARLA, a machine learning-based safety monitoring approach designed for DRL agents. For practical reasons, SMARLA is designed to be black-box (as it does not require access to the internals of the agent) and leverages state abstraction to reduce the state space and thus facilitate the learning of safety violation prediction models from agent's states. We validated SMARLA on two well-known RL case studies. Empirical analysis reveals that SMARLA achieves accurate violation prediction with a low false positive rate, and can predict safety violations at an early stage, approximately halfway through the agent's execution before 
    

