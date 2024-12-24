# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Python Fuzzing for Trustworthy Machine Learning Frameworks](https://arxiv.org/abs/2403.12723) | 提出了一种用于Python项目的动态分析管道，结合模糊测试、语料库最小化、崩溃分类和覆盖率收集，以确保机器学习框架的安全性和可靠性。 |

# 详细

[^1]: 用于可信赖的机器学习框架的Python模糊测试

    Python Fuzzing for Trustworthy Machine Learning Frameworks

    [https://arxiv.org/abs/2403.12723](https://arxiv.org/abs/2403.12723)

    提出了一种用于Python项目的动态分析管道，结合模糊测试、语料库最小化、崩溃分类和覆盖率收集，以确保机器学习框架的安全性和可靠性。

    

    确保机器学习框架的安全性和可靠性对于构建可信赖的基于人工智能的系统至关重要。模糊测试是安全软件开发生命周期（SSDLC）中一种流行的技术，可用于开发安全和健壮的软件。我们提出了使用Sydr-Fuzz工具集针对Python项目的动态分析管道。我们的管道包括模糊测试、语料库最小化、崩溃分类和覆盖率收集。崩溃分类和严重性评估是确保及时解决最关键漏洞的重要步骤。此外，所提出的管道集成在GitLab CI中。为了确定机器学习框架中最易受攻击的部分，我们分析它们潜在的攻击面，并为PyTorch、TensorFlow开发模糊测试目标。

    arXiv:2403.12723v1 Announce Type: cross  Abstract: Ensuring the security and reliability of machine learning frameworks is crucial for building trustworthy AI-based systems. Fuzzing, a popular technique in secure software development lifecycle (SSDLC), can be used to develop secure and robust software. Popular machine learning frameworks such as PyTorch and TensorFlow are complex and written in multiple programming languages including C/C++ and Python. We propose a dynamic analysis pipeline for Python projects using the Sydr-Fuzz toolset. Our pipeline includes fuzzing, corpus minimization, crash triaging, and coverage collection. Crash triaging and severity estimation are important steps to ensure that the most critical vulnerabilities are addressed promptly. Furthermore, the proposed pipeline is integrated in GitLab CI. To identify the most vulnerable parts of the machine learning frameworks, we analyze their potential attack surfaces and develop fuzz targets for PyTorch, TensorFlow, 
    

