# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Instruction Tuning for Secure Code Generation](https://arxiv.org/abs/2402.09497) | 现代语言模型在编程中得到广泛应用，指令调优是一个增强其实用性的关键过程。然而，现有的方案忽视了生成代码的安全性。本文提出了SafeCoder，通过安全微调和标准指令调优相结合，来优化安全性和实用性。 |
| [^2] | [DISTINQT: A Distributed Privacy Aware Learning Framework for QoS Prediction for Future Mobile and Wireless Networks.](http://arxiv.org/abs/2401.10158) | DISTINQT是一种面向未来移动和无线网络的隐私感知分布式学习框架，用于QoS预测。 |

# 详细

[^1]: 安全代码生成的指令调优

    Instruction Tuning for Secure Code Generation

    [https://arxiv.org/abs/2402.09497](https://arxiv.org/abs/2402.09497)

    现代语言模型在编程中得到广泛应用，指令调优是一个增强其实用性的关键过程。然而，现有的方案忽视了生成代码的安全性。本文提出了SafeCoder，通过安全微调和标准指令调优相结合，来优化安全性和实用性。

    

    现代语言模型(LMs)在日常和专业环境中得到了广泛的认可，尤其在编程中。指令调优是一种关键的过程，通过训练LMs遵循用户指令和人类偏好，从而大大增强了LMs的实用性。然而，现有的指令调优方案忽视了一个关键方面：生成代码的安全性。因此，即使是最先进的指令调优的LMs也经常产生不安全的代码，带来了重大的安全风险。在这项工作中，我们引入了SafeCoder来填补这个差距。SafeCoder使用一个多样化和高质量的数据集进行安全为中心的微调，我们使用自动化流水线收集了这个数据集。我们将安全微调与标准的指令调优相结合，以便同时优化安全性和实用性。尽管简单，但我们展示了SafeCoder的有效性。

    arXiv:2402.09497v1 Announce Type: cross  Abstract: Modern language models (LMs) have gained widespread acceptance in everyday and professional contexts, particularly in programming. An essential procedure enabling this adoption is instruction tuning, which substantially enhances LMs' practical utility by training them to follow user instructions and human preferences. However, existing instruction tuning schemes overlook a crucial aspect: the security of generated code. As a result, even the state-of-the-art instruction-tuned LMs frequently produce unsafe code, posing significant security risks. In this work, we introduce SafeCoder to address this gap. SafeCoder performs security-centric fine-tuning using a diverse and high-quality dataset that we collected using an automated pipeline. We integrate the security fine-tuning with standard instruction tuning, to facilitate a joint optimization of both security and utility. Despite its simplicity, we show that SafeCoder is effective across
    
[^2]: DISTINQT: 一种面向未来移动和无线网络的分布式隐私感知学习框架，用于QoS预测

    DISTINQT: A Distributed Privacy Aware Learning Framework for QoS Prediction for Future Mobile and Wireless Networks. (arXiv:2401.10158v1 [cs.NI])

    [http://arxiv.org/abs/2401.10158](http://arxiv.org/abs/2401.10158)

    DISTINQT是一种面向未来移动和无线网络的隐私感知分布式学习框架，用于QoS预测。

    

    5G和6G以后的网络将支持依赖一定服务质量（QoS）的新的和具有挑战性的用例和应用程序。及时预测QoS对于安全关键应用（如车辆通信）尤为重要。尽管直到最近，QoS预测一直由集中式人工智能（AI）解决方案完成，但已经出现了一些隐私、计算和运营方面的问题。替代方案已经出现（如分割学习、联邦学习），将复杂度较低的AI任务分布在节点之间，同时保护数据隐私。然而，考虑到未来无线网络的异构性，当涉及可扩展的分布式学习方法时，会出现新的挑战。该研究提出了一种名为DISTINQT的面向QoS预测的隐私感知分布式学习框架。

    Beyond 5G and 6G networks are expected to support new and challenging use cases and applications that depend on a certain level of Quality of Service (QoS) to operate smoothly. Predicting the QoS in a timely manner is of high importance, especially for safety-critical applications as in the case of vehicular communications. Although until recent years the QoS prediction has been carried out by centralized Artificial Intelligence (AI) solutions, a number of privacy, computational, and operational concerns have emerged. Alternative solutions have been surfaced (e.g. Split Learning, Federated Learning), distributing AI tasks of reduced complexity across nodes, while preserving the privacy of the data. However, new challenges rise when it comes to scalable distributed learning approaches, taking into account the heterogeneous nature of future wireless networks. The current work proposes DISTINQT, a privacy-aware distributed learning framework for QoS prediction. Our framework supports mult
    

