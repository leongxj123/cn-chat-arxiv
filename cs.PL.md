# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM4Decompile: Decompiling Binary Code with Large Language Models](https://arxiv.org/abs/2403.05286) | 发布首批开放访问的反编译LLM，预训练在40亿个C源代码和汇编代码标记上，引入了第一个考虑重新编译性和重新执行性的反编译数据集。 |
| [^2] | [Ansible Lightspeed: A Code Generation Service for IT Automation](https://arxiv.org/abs/2402.17442) | Ansible Lightspeed是一种基于大型语言模型的服务，专注于将自然语言转换为Ansible代码，为IT自动化领域带来了创新。 |
| [^3] | [Explaining Explanations in Probabilistic Logic Programming.](http://arxiv.org/abs/2401.17045) | 该论文介绍了基于概率逻辑编程的解释解释方法，以解决在不透明系统中生成合适解释的困难。 |

# 详细

[^1]: LLM4Decompile：使用大型语言模型对二进制代码进行反编译

    LLM4Decompile: Decompiling Binary Code with Large Language Models

    [https://arxiv.org/abs/2403.05286](https://arxiv.org/abs/2403.05286)

    发布首批开放访问的反编译LLM，预训练在40亿个C源代码和汇编代码标记上，引入了第一个考虑重新编译性和重新执行性的反编译数据集。

    

    反编译旨在将编译代码恢复为可读性强的源代码，但在名称和结构等细节方面存在困难。大型语言模型（LLMs）在编程任务中显示出潜力，激发了它们在反编译中的应用。然而，目前尚无用于反编译的开源LLM。此外，现有的反编译评估系统主要考虑标记级准确性，而很大程度上忽略了代码的可执行性，这是任何程序最重要的特征。因此，我们发布了首批开放访问的反编译LLM，范围从10亿到330亿，预先训练了40亿个令牌的C源代码和相应的汇编代码。这些开源LLM可以作为该领域进一步发展的基线。为了确保实际程序评估，我们引入了Decompile-Eval，这是第一个考虑重新编译性和重新执行性的反编译数据集。该基准强调了评估的重要性。

    arXiv:2403.05286v1 Announce Type: cross  Abstract: Decompilation aims to restore compiled code to human-readable source code, but struggles with details like names and structure. Large language models (LLMs) show promise for programming tasks, motivating their application to decompilation. However, there does not exist any open-source LLM for decompilation. Moreover, existing decompilation evaluation systems mainly consider token-level accuracy and largely ignore code executability, which is the most important feature of any program. Therefore, we release the first open-access decompilation LLMs ranging from 1B to 33B pre-trained on 4 billion tokens of C source code and the corresponding assembly code. The open-source LLMs can serve as baselines for further development in the field. To ensure practical program evaluation, we introduce Decompile-Eval, the first dataset that considers re-compilability and re-executability for decompilation. The benchmark emphasizes the importance of eval
    
[^2]: Ansible Lightspeed: 一种用于IT自动化的代码生成服务

    Ansible Lightspeed: A Code Generation Service for IT Automation

    [https://arxiv.org/abs/2402.17442](https://arxiv.org/abs/2402.17442)

    Ansible Lightspeed是一种基于大型语言模型的服务，专注于将自然语言转换为Ansible代码，为IT自动化领域带来了创新。

    

    大型语言模型（LLMs）的问世使得创建可提高开发者生产力的工具成为可能，集成开发环境（IDEs）常被用作与LLMs交互的接口。已发布许多这类工具，但几乎全部都专注于通用编程语言，很少关注对IT自动化至关重要的特定领域语言。Ansible是一种基于YAML的IT自动化特定语言。Red Hat Ansible Lightspeed与IBM Watson Code Assistant合作的Ansible Lightspeed是一种基于LLM的服务，专门用于将自然语言转换为Ansible代码。

    arXiv:2402.17442v1 Announce Type: cross  Abstract: The availability of Large Language Models (LLMs) which can generate code, has made it possible to create tools that improve developer productivity. Integrated development environments or IDEs which developers use to write software are often used as an interface to interact with LLMs. Although many such tools have been released, almost all of them focus on general-purpose programming languages. Domain-specific languages, such as those crucial for IT automation, have not received much attention. Ansible is one such YAML-based IT automation-specific language. Red Hat Ansible Lightspeed with IBM Watson Code Assistant, further referred to as Ansible Lightspeed, is an LLM-based service designed explicitly for natural language to Ansible code generation.   In this paper, we describe the design and implementation of the Ansible Lightspeed service and analyze feedback from thousands of real users. We examine diverse performance indicators, clas
    
[^3]: 在概率逻辑编程中解释解释

    Explaining Explanations in Probabilistic Logic Programming. (arXiv:2401.17045v1 [cs.AI])

    [http://arxiv.org/abs/2401.17045](http://arxiv.org/abs/2401.17045)

    该论文介绍了基于概率逻辑编程的解释解释方法，以解决在不透明系统中生成合适解释的困难。

    

    基于人工智能的工具的出现也导致了产生人类可理解的解释的需求。在一些方法中，系统是不透明的（通常被称为“黑盒子”），这使得生成适当的解释变得困难。然而，在概率逻辑编程中，我们考虑了逻辑编程（用于知识表示）和概率（用于建模不确定性）的结合。在这个设置中，可以说模型是可以解释的，这方便了对模型的理解。然而，对于特定的查询，通常的“解释”的概念是与模型的每个随机变量的选择集相关联的。不幸的是，这个集合没有因果结构，实际上，一些选择实际上与所考虑的查询无关。为了克服这些缺点，我们提出了一种基于查询驱动推理定义的解释解释方法。

    The emergence of tools based on artificial intelligence has also led to the need of producing explanations which are understandable by a human being. In some approaches, the system is not transparent (often referred to as a "black box"), making it difficult to generate appropriate explanations. In this work, though, we consider probabilistic logic programming, a combination of logic programming (for knowledge representation) and probability (to model uncertainty). In this setting, one can say that models are interpretable, which eases its understanding. However, given a particular query, the usual notion of "explanation" is associated with a set of choices, one for each random variable of the model. Unfortunately, this set does not have a causal structure and, in fact, some of the choices are actually irrelevant to the considered query. In order to overcome these shortcomings, we present an approach to explaining explanations which is based on the definition of a query-driven inference
    

