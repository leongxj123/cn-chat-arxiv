# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Python Fuzzing for Trustworthy Machine Learning Frameworks](https://arxiv.org/abs/2403.12723) | 提出了一种用于Python项目的动态分析管道，结合模糊测试、语料库最小化、崩溃分类和覆盖率收集，以确保机器学习框架的安全性和可靠性。 |
| [^2] | [Leveraging AI Planning For Detecting Cloud Security Vulnerabilities](https://arxiv.org/abs/2402.10985) | 提出了一个通用框架来建模云系统中的访问控制策略，并开发了基于PDDL模型的新方法来检测可能导致诸如勒索软件和敏感数据外泄等广泛攻击的安全漏洞。 |
| [^3] | [Ensembler: Combating model inversion attacks using model ensemble during collaborative inference.](http://arxiv.org/abs/2401.10859) | Ensembler是一个防止模型反演攻击的可扩展框架，通过利用模型集成和引入扰动的方式，在协作推理过程中有效地保护数据隐私。 |

# 详细

[^1]: 用于可信赖的机器学习框架的Python模糊测试

    Python Fuzzing for Trustworthy Machine Learning Frameworks

    [https://arxiv.org/abs/2403.12723](https://arxiv.org/abs/2403.12723)

    提出了一种用于Python项目的动态分析管道，结合模糊测试、语料库最小化、崩溃分类和覆盖率收集，以确保机器学习框架的安全性和可靠性。

    

    确保机器学习框架的安全性和可靠性对于构建可信赖的基于人工智能的系统至关重要。模糊测试是安全软件开发生命周期（SSDLC）中一种流行的技术，可用于开发安全和健壮的软件。我们提出了使用Sydr-Fuzz工具集针对Python项目的动态分析管道。我们的管道包括模糊测试、语料库最小化、崩溃分类和覆盖率收集。崩溃分类和严重性评估是确保及时解决最关键漏洞的重要步骤。此外，所提出的管道集成在GitLab CI中。为了确定机器学习框架中最易受攻击的部分，我们分析它们潜在的攻击面，并为PyTorch、TensorFlow开发模糊测试目标。

    arXiv:2403.12723v1 Announce Type: cross  Abstract: Ensuring the security and reliability of machine learning frameworks is crucial for building trustworthy AI-based systems. Fuzzing, a popular technique in secure software development lifecycle (SSDLC), can be used to develop secure and robust software. Popular machine learning frameworks such as PyTorch and TensorFlow are complex and written in multiple programming languages including C/C++ and Python. We propose a dynamic analysis pipeline for Python projects using the Sydr-Fuzz toolset. Our pipeline includes fuzzing, corpus minimization, crash triaging, and coverage collection. Crash triaging and severity estimation are important steps to ensure that the most critical vulnerabilities are addressed promptly. Furthermore, the proposed pipeline is integrated in GitLab CI. To identify the most vulnerable parts of the machine learning frameworks, we analyze their potential attack surfaces and develop fuzz targets for PyTorch, TensorFlow, 
    
[^2]: 利用AI规划技术检测云安全漏洞

    Leveraging AI Planning For Detecting Cloud Security Vulnerabilities

    [https://arxiv.org/abs/2402.10985](https://arxiv.org/abs/2402.10985)

    提出了一个通用框架来建模云系统中的访问控制策略，并开发了基于PDDL模型的新方法来检测可能导致诸如勒索软件和敏感数据外泄等广泛攻击的安全漏洞。

    

    云计算服务提供了可扩展且具有成本效益的数据存储、处理和协作解决方案。随着它们的普及，与其安全漏洞相关的担忧也在增长，这可能导致数据泄露和勒索软件等复杂攻击。为了应对这些问题，我们首先提出了一个通用框架，用于表达云系统中不同对象（如用户、数据存储、安全角色）之间的关系，以建模云系统中的访问控制策略。访问控制误配置通常是云攻击的主要原因。其次，我们开发了一个PDDL模型，用于检测安全漏洞，例如可能导致广泛攻击（如勒索软件）和敏感数据外泄等。规划器可以生成攻击以识别云中的此类漏洞。最后，我们在14个不同商业组织的真实亚马逊AWS云配置上测试了我们的方法。

    arXiv:2402.10985v1 Announce Type: cross  Abstract: Cloud computing services provide scalable and cost-effective solutions for data storage, processing, and collaboration. Alongside their growing popularity, concerns related to their security vulnerabilities leading to data breaches and sophisticated attacks such as ransomware are growing. To address these, first, we propose a generic framework to express relations between different cloud objects such as users, datastores, security roles, to model access control policies in cloud systems. Access control misconfigurations are often the primary driver for cloud attacks. Second, we develop a PDDL model for detecting security vulnerabilities which can for example lead to widespread attacks such as ransomware, sensitive data exfiltration among others. A planner can then generate attacks to identify such vulnerabilities in the cloud. Finally, we test our approach on 14 real Amazon AWS cloud configurations of different commercial organizations
    
[^3]: Ensembler:使用模型集成在协作推理过程中防止模型反演攻击

    Ensembler: Combating model inversion attacks using model ensemble during collaborative inference. (arXiv:2401.10859v1 [cs.CR])

    [http://arxiv.org/abs/2401.10859](http://arxiv.org/abs/2401.10859)

    Ensembler是一个防止模型反演攻击的可扩展框架，通过利用模型集成和引入扰动的方式，在协作推理过程中有效地保护数据隐私。

    

    深度学习模型在各个领域展示出了卓越的性能。然而，庞大的模型大小促使边缘设备将推理过程的大部分转移到云端。虽然这种做法带来了许多优势，但也引发了关于用户数据隐私的重要问题。在云服务器的可信度受到质疑的情况下，保护数据隐私的实用和适应性方法变得至关重要。在本文中，我们介绍了Ensembler，这是一个可扩展的框架，旨在大大增加对抗方进行模型反演攻击的难度。Ensembler利用在对抗服务器上运行的模型组合，与现有的在协作推理过程中引入扰动到敏感数据的方法并行。我们的实验表明，即使与基本的高斯噪声相结合，Ensembler也可以有效地保护图像免受重建攻击。

    Deep learning models have exhibited remarkable performance across various domains. Nevertheless, the burgeoning model sizes compel edge devices to offload a significant portion of the inference process to the cloud. While this practice offers numerous advantages, it also raises critical concerns regarding user data privacy. In scenarios where the cloud server's trustworthiness is in question, the need for a practical and adaptable method to safeguard data privacy becomes imperative. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks for adversarial parties. Ensembler leverages model ensembling on the adversarial server, running in parallel with existing approaches that introduce perturbations to sensitive data during colloborative inference. Our experiments demonstrate that when combined with even basic Gaussian noise, Ensembler can effectively shield images from reconstruction attacks, 
    

