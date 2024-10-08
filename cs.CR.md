# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^2] | [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2402.17840) | 研究揭示了检索增强生成系统中的数据泄露风险，指出对手可以利用LMs的指示遵循能力轻松地从数据存储中直接提取文本数据，并设计了攻击对生产RAG模型GPTs造成数据存储泄漏。 |
| [^3] | [The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks](https://arxiv.org/abs/2402.06357) | 本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。 |
| [^4] | [Conversation Reconstruction Attack Against GPT Models](https://arxiv.org/abs/2402.02987) | 本文介绍了一种针对 GPT 模型的对话重构攻击，该攻击具有劫持会话和重构对话的两个步骤。通过对该攻击对 GPT 模型的隐私风险进行评估，发现 GPT-4 对该攻击具有一定的鲁棒性。 |
| [^5] | [Dr. Jekyll and Mr. Hyde: Two Faces of LLMs](https://arxiv.org/abs/2312.03853) | 本研究通过让ChatGPT和Bard冒充复杂人物角色，绕过了安全机制和专门训练程序，展示了被禁止的回应实际上被提供了，从而有可能获取未经授权、非法或有害的信息。 |
| [^6] | [FheFL: Fully Homomorphic Encryption Friendly Privacy-Preserving Federated Learning with Byzantine Users.](http://arxiv.org/abs/2306.05112) | 本论文介绍了一种新的联邦学习算法，采用FHE加密技术，既可以保护模型更新的隐私，又可以防止恶意用户破坏全局模型。 |

# 详细

[^1]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^2]: 遵循我的指示并说出真相：来自检索增强生成系统的可扩展数据提取

    Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems

    [https://arxiv.org/abs/2402.17840](https://arxiv.org/abs/2402.17840)

    研究揭示了检索增强生成系统中的数据泄露风险，指出对手可以利用LMs的指示遵循能力轻松地从数据存储中直接提取文本数据，并设计了攻击对生产RAG模型GPTs造成数据存储泄漏。

    

    检索增强生成（RAG）通过在测试时将外部知识纳入预训练模型，从而实现定制适应，提升了模型性能。本研究探讨了Retrieval-In-Context RAG语言模型（LMs）中的数据泄露风险。我们展示了当对使用指令调整的LMs构建的RAG系统进行提示注入时，对手可以利用LMs的指示遵循能力轻松地从数据存储中直接提取文本数据。这种漏洞存在于覆盖Llama2、Mistral/Mixtral、Vicuna、SOLAR、WizardLM、Qwen1.5和Platypus2等多种现代LMs的广泛范围内，并且随着模型规模的扩大，利用能力加剧。将研究扩展到生产RAG模型GPTs，我们设计了一种攻击，可以在对25个随机选择的定制GPTs施加最多2个查询时以100%成功率导致数据存储泄漏，并且我们能够以77,000字的书籍中的文本数据的提取率为41%，以及在含有1,569,00词的语料库中的文本数据的提取率为3%。

    arXiv:2402.17840v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) improves pre-trained models by incorporating external knowledge at test time to enable customized adaptation. We study the risk of datastore leakage in Retrieval-In-Context RAG Language Models (LMs). We show that an adversary can exploit LMs' instruction-following capabilities to easily extract text data verbatim from the datastore of RAG systems built with instruction-tuned LMs via prompt injection. The vulnerability exists for a wide range of modern LMs that span Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, and Platypus2, and the exploitability exacerbates as the model size scales up. Extending our study to production RAG models GPTs, we design an attack that can cause datastore leakage with a 100% success rate on 25 randomly selected customized GPTs with at most 2 queries, and we extract text data verbatim at a rate of 41% from a book of 77,000 words and 3% from a corpus of 1,569,00
    
[^3]: SpongeNet 攻击：深度神经网络的海绵权重中毒

    The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks

    [https://arxiv.org/abs/2402.06357](https://arxiv.org/abs/2402.06357)

    本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。

    

    海绵攻击旨在增加在硬件加速器上部署的神经网络的能耗和计算时间。现有的海绵攻击可以通过海绵示例进行推理，也可以通过海绵中毒在训练过程中进行。海绵示例利用添加到模型输入的扰动来增加能量和延迟，而海绵中毒则改变模型的目标函数来引发推理时的能量/延迟效应。在这项工作中，我们提出了一种新颖的海绵攻击，称为 SpongeNet。SpongeNet 是第一个直接作用于预训练模型参数的海绵攻击。我们的实验表明，相比于海绵中毒，SpongeNet 可以成功增加视觉模型的能耗，并且所需的样本数量更少。我们的实验结果表明，如果不专门针对海绵中毒进行调整（即减小批归一化偏差值），则毒害防御会失效。我们的工作显示出海绵攻击的影响。

    Sponge attacks aim to increase the energy consumption and computation time of neural networks deployed on hardware accelerators. Existing sponge attacks can be performed during inference via sponge examples or during training via Sponge Poisoning. Sponge examples leverage perturbations added to the model's input to increase energy and latency, while Sponge Poisoning alters the objective function of a model to induce inference-time energy/latency effects.   In this work, we propose a novel sponge attack called SpongeNet. SpongeNet is the first sponge attack that is performed directly on the parameters of a pre-trained model. Our experiments show that SpongeNet can successfully increase the energy consumption of vision models with fewer samples required than Sponge Poisoning. Our experiments indicate that poisoning defenses are ineffective if not adjusted specifically for the defense against Sponge Poisoning (i.e., they decrease batch normalization bias values). Our work shows that Spong
    
[^4]: GPT 模型的对话重构攻击

    Conversation Reconstruction Attack Against GPT Models

    [https://arxiv.org/abs/2402.02987](https://arxiv.org/abs/2402.02987)

    本文介绍了一种针对 GPT 模型的对话重构攻击，该攻击具有劫持会话和重构对话的两个步骤。通过对该攻击对 GPT 模型的隐私风险进行评估，发现 GPT-4 对该攻击具有一定的鲁棒性。

    

    最近，在大型语言模型（LLM）领域取得了重要进展，其中 GPT 系列模型代表着最具代表性的成果。为了优化任务执行，用户经常与托管在云环境中的 GPT 模型进行多轮对话。这些多轮对话往往包含私人信息，需要在云中进行传输和存储。然而，这种操作模式引入了额外的攻击面。本文首先介绍了一种针对 GPT 模型的特定对话重构攻击。我们提出的对话重构攻击由两个步骤组成：劫持会话和重构对话。随后，我们对当 GPT 模型遭受该攻击时对话中固有的隐私风险进行了详尽评估。然而，GPT-4 对于该攻击具有一定的鲁棒性。接着，我们引入了两种高级攻击，旨在更好地重构以前的对话。

    In recent times, significant advancements have been made in the field of large language models (LLMs), represented by GPT series models. To optimize task execution, users often engage in multi-round conversations with GPT models hosted in cloud environments. These multi-round conversations, potentially replete with private information, require transmission and storage within the cloud. However, this operational paradigm introduces additional attack surfaces. In this paper, we first introduce a specific Conversation Reconstruction Attack targeting GPT models. Our introduced Conversation Reconstruction Attack is composed of two steps: hijacking a session and reconstructing the conversations. Subsequently, we offer an exhaustive evaluation of the privacy risks inherent in conversations when GPT models are subjected to the proposed attack. However, GPT-4 demonstrates certain robustness to the proposed attacks. We then introduce two advanced attacks aimed at better reconstructing previous c
    
[^5]: LLMs的两面性：Jekyll博士与Hyde先生

    Dr. Jekyll and Mr. Hyde: Two Faces of LLMs

    [https://arxiv.org/abs/2312.03853](https://arxiv.org/abs/2312.03853)

    本研究通过让ChatGPT和Bard冒充复杂人物角色，绕过了安全机制和专门训练程序，展示了被禁止的回应实际上被提供了，从而有可能获取未经授权、非法或有害的信息。

    

    仅仅一年前，我们目睹了大型语言模型（LLMs）的使用增加，尤其是在结合像聊天机器人助手之类的应用时。为了防止这些助手产生不当回应，我们实施了安全机制和专门的训练程序。在这项工作中，我们通过让ChatGPT和Bard（以及在某种程度上是Bing chat）冒充复杂人物角色，绕过了这些措施，这些角色与它们本应成为的真实助手的特征相反。我们首先创造出这些人物角色的复杂传记，然后在同一聊天机器人中使用它们进行新的对话。我们的对话采用角色扮演风格，以获得助手不被允许提供的回应。通过使用人物角色，我们展示了被禁止的回应实际上被提供了，从而有可能获取未经授权、非法或有害的信息。这项工作表明，通过使用对抗性pe

    arXiv:2312.03853v2 Announce Type: replace-cross  Abstract: Only a year ago, we witnessed a rise in the use of Large Language Models (LLMs), especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial pe
    
[^6]: FheFL：支持完全同态加密的隐私保护联邦学习与拜占庭用户

    FheFL: Fully Homomorphic Encryption Friendly Privacy-Preserving Federated Learning with Byzantine Users. (arXiv:2306.05112v1 [cs.AI])

    [http://arxiv.org/abs/2306.05112](http://arxiv.org/abs/2306.05112)

    本论文介绍了一种新的联邦学习算法，采用FHE加密技术，既可以保护模型更新的隐私，又可以防止恶意用户破坏全局模型。

    

    联邦学习（FL）技术最初是为了缓解传统机器学习范式中可能出现的数据隐私问题而开发的。尽管FL确保用户的数据始终保留在用户手中，但局部训练模型的梯度必须与集中式服务器通信以构建全局模型。这导致隐私泄露，使得服务器可以从共享的梯度中推断出用户数据的私密信息。为了缓解这一缺陷，下一代FL架构提出了加密和匿名化技术，以保护模型更新免受服务器的攻击。然而，这种方法会带来其他挑战，例如恶意用户可能通过共享虚假梯度来破坏全局模型。由于梯度被加密，服务器无法识别和排除不良用户以保护全局模型。因此，为了缓解这两种攻击，本文提出了一种基于完全同态加密（FHE）的新方案。

    The federated learning (FL) technique was initially developed to mitigate data privacy issues that can arise in the traditional machine learning paradigm. While FL ensures that a user's data always remain with the user, the gradients of the locally trained models must be communicated with the centralized server to build the global model. This results in privacy leakage, where the server can infer private information of the users' data from the shared gradients. To mitigate this flaw, the next-generation FL architectures proposed encryption and anonymization techniques to protect the model updates from the server. However, this approach creates other challenges, such as a malicious user might sabotage the global model by sharing false gradients. Since the gradients are encrypted, the server is unable to identify and eliminate rogue users which would protect the global model. Therefore, to mitigate both attacks, this paper proposes a novel fully homomorphic encryption (FHE) based scheme 
    

