# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness, Efficiency, or Privacy: Pick Two in Machine Learning](https://arxiv.org/abs/2312.14712) | 该论文研究了在分布式机器学习架构中实现隐私和鲁棒性的成本，指出整合这两个目标会牺牲计算效率。 |
| [^2] | [A Novel Federated Learning-Based IDS for Enhancing UAVs Privacy and Security](https://arxiv.org/abs/2312.04135) | 本论文引入了基于联邦学习的入侵检测系统(FL-IDS)，旨在解决FANETs中集中式系统所遇到的挑战，降低了计算和存储成本，适合资源受限的无人机。 |
| [^3] | [PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks.](http://arxiv.org/abs/2401.10586) | PuriDefense是一种高效的防御机制，通过使用轻量级净化模型进行随机路径净化，减缓基于查询的攻击的收敛速度，并有效防御黑盒基于查询的攻击。 |

# 详细

[^1]: 机器学习中的鲁棒性、效率或隐私：只能选两样

    Robustness, Efficiency, or Privacy: Pick Two in Machine Learning

    [https://arxiv.org/abs/2312.14712](https://arxiv.org/abs/2312.14712)

    该论文研究了在分布式机器学习架构中实现隐私和鲁棒性的成本，指出整合这两个目标会牺牲计算效率。

    

    机器学习（ML）应用的成功依赖于庞大的数据集和分布式架构，随着它们的增长，这些架构带来了重大挑战。在真实世界的场景中，数据通常包含敏感信息，数据污染和硬件故障等问题很常见。确保隐私和鲁棒性对于ML在公共生活中的广泛应用至关重要。本文从理论和实证角度研究了在分布式ML架构中实现这些目标所带来的成本。我们概述了分布式ML中隐私和鲁棒性的含义，并阐明了如何单独高效实现它们。然而，我们认为整合这两个目标会在计算效率上有显著的折衷。简而言之，传统的噪声注入通过隐藏毒害输入来损害准确性，而加密方法与防毒防御相冲突，因为它们是非线性的。

    arXiv:2312.14712v2 Announce Type: replace  Abstract: The success of machine learning (ML) applications relies on vast datasets and distributed architectures which, as they grow, present major challenges. In real-world scenarios, where data often contains sensitive information, issues like data poisoning and hardware failures are common. Ensuring privacy and robustness is vital for the broad adoption of ML in public life. This paper examines the costs associated with achieving these objectives in distributed ML architectures, from both theoretical and empirical perspectives. We overview the meanings of privacy and robustness in distributed ML, and clarify how they can be achieved efficiently in isolation. However, we contend that the integration of these two objectives entails a notable compromise in computational efficiency. In short, traditional noise injection hurts accuracy by concealing poisoned inputs, while cryptographic methods clash with poisoning defenses due to their non-line
    
[^2]: 基于联邦学习的用于增强无人机隐私和安全的入侵检测系统

    A Novel Federated Learning-Based IDS for Enhancing UAVs Privacy and Security

    [https://arxiv.org/abs/2312.04135](https://arxiv.org/abs/2312.04135)

    本论文引入了基于联邦学习的入侵检测系统(FL-IDS)，旨在解决FANETs中集中式系统所遇到的挑战，降低了计算和存储成本，适合资源受限的无人机。

    

    无人机在飞行自组织网络(FANETs)中运行时会遇到安全挑战，因为这些网络具有动态和分布式的特性。先前的研究主要集中在集中式入侵检测上，假设一个中央实体负责存储和分析来自所有设备的数据。然而，这些方法面临计算和存储成本以及单点故障风险等挑战，威胁到数据隐私和可用性。数据在互连设备之间广泛分散的情况突显了去中心化方法的必要性。本文介绍了基于联邦学习的入侵检测系统(FL-IDS)，解决了FANETs中集中式系统遇到的挑战。FL-IDS在去中心化方式下运行，降低了客户端和中央服务器的计算和存储成本，这对于资源受限的无人机至关重要。

    arXiv:2312.04135v2 Announce Type: replace-cross  Abstract: Unmanned aerial vehicles (UAVs) operating within Flying Ad-hoc Networks (FANETs) encounter security challenges due to the dynamic and distributed nature of these networks. Previous studies predominantly focused on centralized intrusion detection, assuming a central entity responsible for storing and analyzing data from all devices.However, these approaches face challenges including computation and storage costs, along with a single point of failure risk, threatening data privacy and availability. The widespread dispersion of data across interconnected devices underscores the necessity for decentralized approaches. This paper introduces the Federated Learning-based Intrusion Detection System (FL-IDS), addressing challenges encountered by centralized systems in FANETs. FL-IDS reduces computation and storage costs for both clients and the central server, crucial for resource-constrained UAVs. Operating in a decentralized manner, F
    
[^3]: PuriDefense：用于防御黑盒基于查询的攻击的随机局部隐式对抗净化

    PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks. (arXiv:2401.10586v1 [cs.CR])

    [http://arxiv.org/abs/2401.10586](http://arxiv.org/abs/2401.10586)

    PuriDefense是一种高效的防御机制，通过使用轻量级净化模型进行随机路径净化，减缓基于查询的攻击的收敛速度，并有效防御黑盒基于查询的攻击。

    

    黑盒基于查询的攻击对机器学习作为服务系统构成重大威胁，因为它们可以生成对抗样本而不需要访问目标模型的架构和参数。传统的防御机制，如对抗训练、梯度掩盖和输入转换，要么带来巨大的计算成本，要么损害非对抗输入的测试准确性。为了应对这些挑战，我们提出了一种高效的防御机制PuriDefense，在低推理成本的级别上使用轻量级净化模型的随机路径净化。这些模型利用局部隐式函数并重建自然图像流形。我们的理论分析表明，这种方法通过将随机性纳入净化过程来减缓基于查询的攻击的收敛速度。对CIFAR-10和ImageNet的大量实验验证了我们提出的净化器防御的有效性。

    Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defen
    

