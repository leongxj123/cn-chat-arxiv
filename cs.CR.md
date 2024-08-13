# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models](https://arxiv.org/abs/2403.15740) | 通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。 |
| [^2] | [PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2402.07867) | 本论文提出了一种名为PoisonedRAG的知识污染攻击方法，用于对大型语言模型的检索增强生成进行攻击和破坏。 |
| [^3] | [Private Fine-tuning of Large Language Models with Zeroth-order Optimization.](http://arxiv.org/abs/2401.04343) | 引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。 |

# 详细

[^1]: Ghost Sentence：一种供普通用户使用的工具，用于对大型语言模型中的数据进行版权保护

    Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models

    [https://arxiv.org/abs/2403.15740](https://arxiv.org/abs/2403.15740)

    通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。

    

    Web用户数据在预训练大型语言模型（LLMs）及其微调变种的生态系统中起着核心作用。本文提出了一种方法，建议用户在其文档中反复插入个人密码，使LLMs能够记忆这些密码。这些用户文档中隐藏的密码，被称为“幽灵句子”，一旦它们出现在LLMs生成的内容中，用户就可以确信他们的数据被用于训练。为了探索这种版权工具的有效性和用法，我们利用幽灵句子定义了“用户训练数据识别”任务。我们创建了来自不同来源、不同规模的多个数据集，并使用不同规模的LLMs进行测试。为了评估，我们引入了一个最后$k$个单词验证的方式。

    arXiv:2403.15740v1 Announce Type: new  Abstract: Web user data plays a central role in the ecosystem of pre-trained large language models (LLMs) and their fine-tuned variants. Billions of data are crawled from the web and fed to LLMs. How can \textit{\textbf{everyday web users}} confirm if LLMs misuse their data without permission? In this work, we suggest that users repeatedly insert personal passphrases into their documents, enabling LLMs to memorize them. These concealed passphrases in user documents, referred to as \textit{ghost sentences}, once they are identified in the generated content of LLMs, users can be sure that their data is used for training. To explore the effectiveness and usage of this copyrighting tool, we define the \textit{user training data identification} task with ghost sentences. Multiple datasets from various sources at different scales are created and tested with LLMs of different sizes. For evaluation, we introduce a last $k$ words verification manner along 
    
[^2]: PoisonedRAG: 知识污染攻击对大型语言模型的检索增强生成

    PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models

    [https://arxiv.org/abs/2402.07867](https://arxiv.org/abs/2402.07867)

    本论文提出了一种名为PoisonedRAG的知识污染攻击方法，用于对大型语言模型的检索增强生成进行攻击和破坏。

    

    大型语言模型（LLM）由于其卓越的生成能力而取得了显著的成功。尽管如此，它们也存在固有的局限性，如缺乏最新的知识和虚构。检索增强生成（RAG）是一种最先进的技术，以减轻这些限制。具体而言，对于给定的问题，RAG从知识数据库中检索相关知识，以增强LLM的输入。例如，当知识数据库中包含从维基百科收集的数百万个文本时，检索到的知识可以是与给定问题在语义上最相似的前K个文本集。因此，LLM可以利用检索到的知识作为上下文为给定问题生成答案。现有研究主要集中在改善RAG的准确性或效率，而对其安全性的探索较少。我们旨在填补这一空白。

    Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate those limitations. In particular, given a question, RAG retrieves relevant knowledge from a knowledge database to augment the input of the LLM. For instance, the retrieved knowledge could be a set of top-k texts that are most semantically similar to the given question when the knowledge database contains millions of texts collected from Wikipedia. As a result, the LLM could utilize the retrieved knowledge as the context to generate an answer for the given question. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. Particularly, we propose PoisonedRAG , a set of knowledge pois
    
[^3]: 私有零阶优化的大型语言模型的私有微调

    Private Fine-tuning of Large Language Models with Zeroth-order Optimization. (arXiv:2401.04343v1 [cs.LG])

    [http://arxiv.org/abs/2401.04343](http://arxiv.org/abs/2401.04343)

    引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。

    

    在私有数据集上对大型预训练模型进行微调可能会存在违反隐私的风险。差分隐私是一种通过强制算法稳定性来减轻隐私风险的框架。DP-SGD可以以保护隐私的方式训练具有私有数据的模型，但会带来性能损失和重大工程挑战。我们引入了DP-ZO，一种通过私有化零阶优化来保护训练数据隐私的大型语言模型微调方法。我们的方法设计的一个关键见解是，我们使用的零阶算法SPSA中的梯度方向始终是随机的，而仅依赖于私有数据的信息是步长，即一个标量。因此，我们只需要对标量步长进行隐私处理，这是存储效率高的方法。DP-ZO可以使用拉普拉斯噪声或高斯噪声来实现，在不同任务之间提供了隐私和效用之间的强大权衡。

    Fine-tuning large pretrained models on private datasets may run the risk of violating privacy. Differential privacy is a framework for mitigating privacy risks by enforcing algorithmic stability. DP-SGD enables training models with private data in a privacy-preserving manner, but raises new obstacles in the form of performance loss and significant engineering challenges. We introduce DP-ZO, a new method for fine-tuning large language models that preserves the privacy of training data by privatizing zeroth-order optimization. A key insight into the design of our method is that the direction of the gradient in SPSA, the zeroth-order algorithm we use, is always random and the only information that depends on private data is the step size, i.e., a scalar. Therefore, we only need to privatize the scalar step size, which is memory-efficient. DP-ZO, which can be instantiated with either Laplace or Gaussian noise, provides a strong privacy-utility trade-off across different tasks, and model si
    

