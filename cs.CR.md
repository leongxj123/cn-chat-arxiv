# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) | 提出了一种新的基准 StrongREJECT，通过使用更高质量的问题，更好地区分有效和无效的空破解方法。 |
| [^2] | [When Fairness Meets Privacy: Exploring Privacy Threats in Fair Binary Classifiers through Membership Inference Attacks.](http://arxiv.org/abs/2311.03865) | 本研究探索了公平二分类器中的隐私威胁，并揭示了针对公平增强模型的基于分数的成员推断攻击的无效性。同时，公平性方法可能导致训练数据中大多数子群体的预测性能下降。 |
| [^3] | [Split-and-Denoise: Protect large language model inference with local differential privacy.](http://arxiv.org/abs/2310.09130) | 本文提出了一种名为SnD的创新框架，用于保护大型语言模型推断阶段的隐私。该方法通过在客户端上执行令牌嵌入层和引入噪声来优化隐私-效用权衡，无需修改模型参数。 |

# 详细

[^1]: 一种用于空破解的强REJECT方法

    A StrongREJECT for Empty Jailbreaks

    [https://arxiv.org/abs/2402.10260](https://arxiv.org/abs/2402.10260)

    提出了一种新的基准 StrongREJECT，通过使用更高质量的问题，更好地区分有效和无效的空破解方法。

    

    大型语言模型（LLMs）的兴起引起了对“破解”的关注，这种破解允许模型被恶意使用。然而，目前没有标准的基准来衡量破解的严重程度，导致破解论文的作者不得不自行创建标准。我们表明这些基准经常包含模棱两可或无法回答的问题，并使用倾向于高估低质量模型响应的滥用潜力的评分标准。一些破解技术使问题更加严重，因为它们即使对于良性问题也会降低模型响应的质量：我们展示了几种破解技术显着降低了GPT-4在MMLU上的零射击表现。破解还会使从“未经审查”的开源模型中获取有害响应变得更加困难。我们提出了一个新的基准，StrongREJECT，通过使用更高质量的问题更好地区分有效和无效的破解方法。

    arXiv:2402.10260v1 Announce Type: cross  Abstract: The rise of large language models (LLMs) has drawn attention to the existence of "jailbreaks" that allow the models to be used maliciously. However, there is no standard benchmark for measuring the severity of a jailbreak, leaving authors of jailbreak papers to create their own. We show that these benchmarks often include vague or unanswerable questions and use grading criteria that are biased towards overestimating the misuse potential of low-quality model responses. Some jailbreak techniques make the problem worse by decreasing the quality of model responses even on benign questions: we show that several jailbreaking techniques substantially reduce the zero-shot performance of GPT-4 on MMLU. Jailbreaks can also make it harder to elicit harmful responses from an "uncensored" open-source model. We present a new benchmark, StrongREJECT, which better discriminates between effective and ineffective jailbreaks by using a higher-quality que
    
[^2]: 当公平性遇见隐私：通过成员推断攻击探索公平二分类器中的隐私威胁

    When Fairness Meets Privacy: Exploring Privacy Threats in Fair Binary Classifiers through Membership Inference Attacks. (arXiv:2311.03865v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.03865](http://arxiv.org/abs/2311.03865)

    本研究探索了公平二分类器中的隐私威胁，并揭示了针对公平增强模型的基于分数的成员推断攻击的无效性。同时，公平性方法可能导致训练数据中大多数子群体的预测性能下降。

    

    先前的研究开发了针对具有歧视行为的有偏模型的公平方法，以达到公平预测的目标。然而，最近的研究发现这些模型在基于分数的成员推断攻击中存在潜在的漏洞。在这些攻击中，对手可以通过分析模型的预测分数推断出特定数据样本是否在训练过程中使用。然而，我们的调查发现，针对公平增强模型的基于分数的成员推断攻击是无效的。针对这些攻击训练的模型退化为简单的阈值模型，从而导致攻击性能降低。与此同时，我们观察到公平性方法往往导致训练数据中的大多数子群体的预测性能下降。这提高了成功攻击的难度，同时扩大了成员和非成员数据之间的预测差距。

    Previous studies have developed fairness methods for biased models that exhibit discriminatory behaviors towards specific subgroups. While these models have shown promise in achieving fair predictions, recent research has identified their potential vulnerability to score-based membership inference attacks (MIAs). In these attacks, adversaries can infer whether a particular data sample was used during training by analyzing the model's prediction scores. However, our investigations reveal that these score-based MIAs are ineffective when targeting fairness-enhanced models in binary classifications. The attack models trained to launch the MIAs degrade into simplistic threshold models, resulting in lower attack performance. Meanwhile, we observe that fairness methods often lead to prediction performance degradation for the majority subgroups of the training data. This raises the barrier to successful attacks and widens the prediction gaps between member and non-member data. Building upon th
    
[^3]: 使用本地差分隐私保护大型语言模型推断：拆分与去噪

    Split-and-Denoise: Protect large language model inference with local differential privacy. (arXiv:2310.09130v1 [cs.AI])

    [http://arxiv.org/abs/2310.09130](http://arxiv.org/abs/2310.09130)

    本文提出了一种名为SnD的创新框架，用于保护大型语言模型推断阶段的隐私。该方法通过在客户端上执行令牌嵌入层和引入噪声来优化隐私-效用权衡，无需修改模型参数。

    

    大型语言模型（LLMs）通过捕捉向量空间中的隐藏语义，展示了在自然语言理解方面的强大能力。这一过程丰富了文本嵌入的价值，从而促进了作为服务（EaaS）的嵌入模型商业模式。然而，直接将文本传输到服务器面临着较大的隐私泄露风险，这是一个尚未得到有效解决的问题。为了缓解这个问题，我们引入了Split-N-Denoise（SnD），一种创新的框架，通过在客户端上以最小的计算成本执行令牌嵌入层来拆分模型。这使得客户端能够在将嵌入传输到服务器之前引入噪声，并随后接收和去噪后的扰动输出嵌入用于下游任务。我们的方法专为LLMs的推断阶段设计，不需要修改模型参数。广泛的实验证明了SnD在各种LLM中优化隐私-效用权衡方面的有效性。

    Large Language Models (LLMs) shows powerful capability in natural language understanding by capturing hidden semantics in vector space. This process enriches the value of the text embeddings for various downstream tasks, thereby fostering the Embedding-as-a-Service (EaaS) business model. However, the direct transmission of text to servers poses a largely unaddressed risk of privacy leakage. To mitigate this issue, we introduce Split-N-Denoise (SnD), an innovative framework that split the model to execute the token embedding layer on the client side at minimal computational cost. This allows the client to introduce noise prior to transmitting the embeddings to the server, and subsequently receive and denoise the perturbed output embeddings for downstream tasks. Our approach is designed for the inference stage of LLMs and requires no modifications to the model parameters. Extensive experiments demonstrate SnD's effectiveness in optimizing the privacy-utility tradeoff across various LLM a
    

