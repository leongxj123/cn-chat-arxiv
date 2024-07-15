# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Instruction Tuning for Secure Code Generation](https://arxiv.org/abs/2402.09497) | 现代语言模型在编程中得到广泛应用，指令调优是一个增强其实用性的关键过程。然而，现有的方案忽视了生成代码的安全性。本文提出了SafeCoder，通过安全微调和标准指令调优相结合，来优化安全性和实用性。 |
| [^2] | [Boundary State Generation for Testing and Improvement of Autonomous Driving Systems.](http://arxiv.org/abs/2307.10590) | 该论文介绍了一种新的自动驾驶系统测试生成器（GenBo），它通过在无故障环境中变异自动驾驶车辆的驾驶条件来生成边界状态对，以解决现有测试方法中存在的问题。 |

# 详细

[^1]: 安全代码生成的指令调优

    Instruction Tuning for Secure Code Generation

    [https://arxiv.org/abs/2402.09497](https://arxiv.org/abs/2402.09497)

    现代语言模型在编程中得到广泛应用，指令调优是一个增强其实用性的关键过程。然而，现有的方案忽视了生成代码的安全性。本文提出了SafeCoder，通过安全微调和标准指令调优相结合，来优化安全性和实用性。

    

    现代语言模型(LMs)在日常和专业环境中得到了广泛的认可，尤其在编程中。指令调优是一种关键的过程，通过训练LMs遵循用户指令和人类偏好，从而大大增强了LMs的实用性。然而，现有的指令调优方案忽视了一个关键方面：生成代码的安全性。因此，即使是最先进的指令调优的LMs也经常产生不安全的代码，带来了重大的安全风险。在这项工作中，我们引入了SafeCoder来填补这个差距。SafeCoder使用一个多样化和高质量的数据集进行安全为中心的微调，我们使用自动化流水线收集了这个数据集。我们将安全微调与标准的指令调优相结合，以便同时优化安全性和实用性。尽管简单，但我们展示了SafeCoder的有效性。

    arXiv:2402.09497v1 Announce Type: cross  Abstract: Modern language models (LMs) have gained widespread acceptance in everyday and professional contexts, particularly in programming. An essential procedure enabling this adoption is instruction tuning, which substantially enhances LMs' practical utility by training them to follow user instructions and human preferences. However, existing instruction tuning schemes overlook a crucial aspect: the security of generated code. As a result, even the state-of-the-art instruction-tuned LMs frequently produce unsafe code, posing significant security risks. In this work, we introduce SafeCoder to address this gap. SafeCoder performs security-centric fine-tuning using a diverse and high-quality dataset that we collected using an automated pipeline. We integrate the security fine-tuning with standard instruction tuning, to facilitate a joint optimization of both security and utility. Despite its simplicity, we show that SafeCoder is effective across
    
[^2]: 自动驾驶系统测试与改进的边界状态生成

    Boundary State Generation for Testing and Improvement of Autonomous Driving Systems. (arXiv:2307.10590v1 [cs.SE])

    [http://arxiv.org/abs/2307.10590](http://arxiv.org/abs/2307.10590)

    该论文介绍了一种新的自动驾驶系统测试生成器（GenBo），它通过在无故障环境中变异自动驾驶车辆的驾驶条件来生成边界状态对，以解决现有测试方法中存在的问题。

    

    最近深度神经网络（DNN）和传感器技术的进展使得自动驾驶系统（ADS）具有了越来越高的自主性。然而，评估其可靠性仍然是一个关键问题。目前的ADS测试方法修改模拟驾驶环境的可控属性，直到ADS出现问题。这种方法有两个主要缺点：（1）对模拟环境的修改可能不容易转移到实际测试环境（例如改变道路形状）；（2）即使ADS在某些环境中成功，这些环境实例也会被丢弃，尽管它们可能包含ADS可能出现问题的潜在驾驶条件。本文提出了一种新的ADS测试生成器——GenBo（GENerator of BOundary state pairs）。GenBo在一个无故障环境实例中变异自动驾驶车辆的驾驶条件（位置，速度和方向），并有效地生成可边界化的状态对。

    Recent advances in Deep Neural Networks (DNNs) and sensor technologies are enabling autonomous driving systems (ADSs) with an ever-increasing level of autonomy. However, assessing their dependability remains a critical concern. State-of-the-art ADS testing approaches modify the controllable attributes of a simulated driving environment until the ADS misbehaves. Such approaches have two main drawbacks: (1) modifications to the simulated environment might not be easily transferable to the in-field test setting (e.g., changing the road shape); (2) environment instances in which the ADS is successful are discarded, despite the possibility that they could contain hidden driving conditions in which the ADS may misbehave.  In this paper, we present GenBo (GENerator of BOundary state pairs), a novel test generator for ADS testing. GenBo mutates the driving conditions of the ego vehicle (position, velocity and orientation), collected in a failure-free environment instance, and efficiently gener
    

