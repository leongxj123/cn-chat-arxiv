# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs](https://arxiv.org/abs/2403.15676) | 该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。 |
| [^2] | [CodeTF: One-stop Transformer Library for State-of-the-art Code LLM.](http://arxiv.org/abs/2306.00029) | CodeTF是一个开源的Transformer库，提供了包括预训练的Code LLM模型和标准化接口等一系列功能，可以轻松地将最先进的Code LLM模型应用于各种软件工程任务中。 |

# 详细

[^1]: AC4：用于ZKP中电路约束的代数计算检查器

    AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs

    [https://arxiv.org/abs/2403.15676](https://arxiv.org/abs/2403.15676)

    该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。

    

    ZKP系统已经引起了人们的关注，在当代密码学中发挥着基础性作用。 Zk-SNARK协议主导了ZKP的使用，通常通过算术电路编程范式实现。然而，欠约束或过约束的电路可能导致错误。 欠约束的电路指的是缺乏必要约束的电路，导致电路中出现意外解决方案，并导致验证者接受错误见证。 过约束的电路是指约束过度的电路，导致电路缺乏必要的解决方案，并导致验证者接受没有见证，使电路毫无意义。 本文介绍了一种新方法，用于找出ZKP电路中两种不同类型的错误。 该方法涉及将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统。

    arXiv:2403.15676v1 Announce Type: cross  Abstract: ZKP systems have surged attention and held a fundamental role in contemporary cryptography. Zk-SNARK protocols dominate the ZKP usage, often implemented through arithmetic circuit programming paradigm. However, underconstrained or overconstrained circuits may lead to bugs. Underconstrained circuits refer to circuits that lack the necessary constraints, resulting in unexpected solutions in the circuit and causing the verifier to accept a bogus witness. Overconstrained circuits refer to circuits that are constrained excessively, resulting in the circuit lacking necessary solutions and causing the verifier to accept no witness, rendering the circuit meaningless. This paper introduces a novel approach for pinpointing two distinct types of bugs in ZKP circuits. The method involves encoding the arithmetic circuit constraints to polynomial equation systems and solving polynomial equation systems over a finite field by algebraic computation. T
    
[^2]: CodeTF：一站式Transformer库，实现最先进的代码LLM

    CodeTF: One-stop Transformer Library for State-of-the-art Code LLM. (arXiv:2306.00029v1 [cs.SE])

    [http://arxiv.org/abs/2306.00029](http://arxiv.org/abs/2306.00029)

    CodeTF是一个开源的Transformer库，提供了包括预训练的Code LLM模型和标准化接口等一系列功能，可以轻松地将最先进的Code LLM模型应用于各种软件工程任务中。

    

    代码智能在转型现代软件工程中扮演着重要角色。近年来，基于深度学习的模型，尤其是利用大量开源代码和编程语言特征的Transformer-based大型语言模型（LLMs），已经展示出了对这些任务的显著潜力。然而，这些模型的开发和部署通常需要对机器学习和软件工程的专业知识，从而为模型应用带来了一定的障碍。本文提出了CodeTF，一个基于Transformer的开放源代码库，用于实现最先进的Code LLM和代码智能。我们采用模块化设计和可扩展框架的原则，设计CodeTF并提供统一接口，以便快速访问和开发不同类型的模型、数据集和任务。我们的库支持预训练的Code LLM模型和流行的代码基准测试，包括标准化接口以有效地训练和服务代码LLMs，并支持双GPU训练和推理。使用CodeTF，用户可以轻松将最先进的Code LLM模型应用于各种软件工程任务中，减少训练工作量。

    Code intelligence plays a key role in transforming modern software engineering. Recently, deep learning-based models, especially Transformer-based large language models (LLMs), have demonstrated remarkable potential in tackling these tasks by leveraging massive open-source code data and programming language features. However, the development and deployment of such models often require expertise in both machine learning and software engineering, creating a barrier for the model adoption. In this paper, we present CodeTF, an open-source Transformer-based library for state-of-the-art Code LLMs and code intelligence. Following the principles of modular design and extensible framework, we design CodeTF with a unified interface to enable rapid access and development across different types of models, datasets and tasks. Our library supports a collection of pretrained Code LLM models and popular code benchmarks, including a standardized interface to train and serve code LLMs efficiently, and d
    

