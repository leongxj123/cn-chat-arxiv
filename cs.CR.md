# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904) | 本文研究了LLM生成文本的放射性，表明使用数字水印训练数据能更容易检测到，同时也展示了即使只有很少比例的水印训练文本，仍可以高置信度地检测出使用数字水印进行微调的情况。 |
| [^2] | [PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining](https://arxiv.org/abs/2402.09477) | PANORAMIA是一种无需重新训练的机器学习模型隐私审计方案，通过使用生成的“非成员”数据进行成员推断攻击，可以量化大规模ML模型的隐私泄露，而无需控制训练过程或重新训练模型，只需要访问训练数据的子集。 |
| [^3] | [On Differentially Private Subspace Estimation Without Distributional Assumptions](https://arxiv.org/abs/2402.06465) | 本论文研究了在没有分布假设的情况下，差分隐私子空间估计的问题。通过使用少量的数据点，可以私密地识别出低维结构，避免了高维度的代价。 |
| [^4] | [Adversarial Robustness Through Artifact Design](https://arxiv.org/abs/2402.04660) | 该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。 |
| [^5] | [Teach Large Language Models to Forget Privacy.](http://arxiv.org/abs/2401.00870) | 这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。 |
| [^6] | [Nebula: Self-Attention for Dynamic Malware Analysis.](http://arxiv.org/abs/2310.10664) | Nebula是一个自注意力网络，用于动态分析恶意软件。它能够概括不同的行为表示和格式，并结合动态日志报告中的异构信息。实验证明Nebula在三个重要任务上表现出色。 |

# 详细

[^1]: 数字水印使语言模型具有放射性

    Watermarking Makes Language Models Radioactive

    [https://arxiv.org/abs/2402.14904](https://arxiv.org/abs/2402.14904)

    本文研究了LLM生成文本的放射性，表明使用数字水印训练数据能更容易检测到，同时也展示了即使只有很少比例的水印训练文本，仍可以高置信度地检测出使用数字水印进行微调的情况。

    

    本文研究了LLM生成的文本的放射性，即是否可以检测到这种输入被用作训练数据。传统方法如成员推断可以以一定水平的准确性进行这种检测。我们表明，带有数字水印的训练数据留下的痕迹比成员推断更容易检测且更可靠。我们将污染水平与水印的鲁棒性、在训练集中的比例和微调过程联系起来。特别是我们展示，即使只有5％的训练文本被数字水印标记，训练在带有数字水印的合成指令上仍然可以具有高置信度（p值<1e-5）被检测到。因此，原本设计用于检测机器生成文本的LLM水印技术，使我们能够轻松确定带有数字水印的LLM的输出是否被用来对另一个LLM进行微调。

    arXiv:2402.14904v1 Announce Type: cross  Abstract: This paper investigates the radioactivity of LLM-generated texts, i.e. whether it is possible to detect that such input was used as training data. Conventional methods like membership inference can carry out this detection with some level of accuracy. We show that watermarked training data leaves traces easier to detect and much more reliable than membership inference. We link the contamination level to the watermark robustness, its proportion in the training set, and the fine-tuning process. We notably demonstrate that training on watermarked synthetic instructions can be detected with high confidence (p-value < 1e-5) even when as little as 5% of training text is watermarked. Thus, LLM watermarking, originally designed for detecting machine-generated text, gives the ability to easily identify if the outputs of a watermarked LLM were used to fine-tune another LLM.
    
[^2]: PANORAMIA: 无需重新训练的机器学习模型隐私审计

    PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining

    [https://arxiv.org/abs/2402.09477](https://arxiv.org/abs/2402.09477)

    PANORAMIA是一种无需重新训练的机器学习模型隐私审计方案，通过使用生成的“非成员”数据进行成员推断攻击，可以量化大规模ML模型的隐私泄露，而无需控制训练过程或重新训练模型，只需要访问训练数据的子集。

    

    我们引入了一种隐私审计方案，该方案依赖于使用生成的“非成员”数据进行成员推断攻击来对ML模型进行隐私审计。这个方案被称为PANORAMIA，它可以量化大规模ML模型的隐私泄露，而无需控制训练过程或重新训练模型，只需要访问训练数据的子集。为了证明其适用性，我们在多个ML领域进行了审计，包括图像和表格数据分类以及大规模语言模型。

    arXiv:2402.09477v1 Announce Type: cross  Abstract: We introduce a privacy auditing scheme for ML models that relies on membership inference attacks using generated data as "non-members". This scheme, which we call PANORAMIA, quantifies the privacy leakage for large-scale ML models without control of the training process or model re-training and only requires access to a subset of the training data. To demonstrate its applicability, we evaluate our auditing scheme across multiple ML domains, ranging from image and tabular data classification to large-scale language models.
    
[^3]: 关于无分布假设的差分隐私子空间估计

    On Differentially Private Subspace Estimation Without Distributional Assumptions

    [https://arxiv.org/abs/2402.06465](https://arxiv.org/abs/2402.06465)

    本论文研究了在没有分布假设的情况下，差分隐私子空间估计的问题。通过使用少量的数据点，可以私密地识别出低维结构，避免了高维度的代价。

    

    隐私数据分析面临着一个被称为维数诅咒的重大挑战，导致了成本的增加。然而，许多数据集具有固有的低维结构。例如，在梯度下降优化过程中，梯度经常位于一个低维子空间附近。如果可以使用少量点私密地识别出这种低维结构，就可以避免因高维度而支付隐私和准确性的代价。

    Private data analysis faces a significant challenge known as the curse of dimensionality, leading to increased costs. However, many datasets possess an inherent low-dimensional structure. For instance, during optimization via gradient descent, the gradients frequently reside near a low-dimensional subspace. If the low-dimensional structure could be privately identified using a small amount of points, we could avoid paying (in terms of privacy and accuracy) for the high ambient dimension.   On the negative side, Dwork, Talwar, Thakurta, and Zhang (STOC 2014) proved that privately estimating subspaces, in general, requires an amount of points that depends on the dimension. But Singhal and Steinke (NeurIPS 2021) bypassed this limitation by considering points that are i.i.d. samples from a Gaussian distribution whose covariance matrix has a certain eigenvalue gap. Yet, it was still left unclear whether we could provide similar upper bounds without distributional assumptions and whether we 
    
[^4]: 通过艺术设计提高对抗性鲁棒性

    Adversarial Robustness Through Artifact Design

    [https://arxiv.org/abs/2402.04660](https://arxiv.org/abs/2402.04660)

    该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。

    

    对抗性示例的出现给机器学习带来了挑战。为了阻碍对抗性示例，大多数防御方法都改变了模型的训练方式（如对抗性训练）或推理过程（如随机平滑）。尽管这些方法显著提高了模型的对抗性鲁棒性，但模型仍然极易受到对抗性示例的影响。在某些领域如交通标志识别中，我们发现对象是按照规范来设计（如标志规范）。为了改善对抗性鲁棒性，我们提出了一种新颖的方法。具体来说，我们提供了一种重新定义规范的方法，对现有规范进行微小的更改，以防御对抗性示例。我们将艺术设计问题建模为一个鲁棒优化问题，并提出了基于梯度和贪婪搜索的方法来解决它。我们在交通标志识别领域对我们的方法进行了评估，使其能够改变交通标志中的象形图标（即标志内的符号）。

    Adversarial examples arose as a challenge for machine learning. To hinder them, most defenses alter how models are trained (e.g., adversarial training) or inference is made (e.g., randomized smoothing). Still, while these approaches markedly improve models' adversarial robustness, models remain highly susceptible to adversarial examples. Identifying that, in certain domains such as traffic-sign recognition, objects are implemented per standards specifying how artifacts (e.g., signs) should be designed, we propose a novel approach for improving adversarial robustness. Specifically, we offer a method to redefine standards, making minor changes to existing ones, to defend against adversarial examples. We formulate the problem of artifact design as a robust optimization problem, and propose gradient-based and greedy search methods to solve it. We evaluated our approach in the domain of traffic-sign recognition, allowing it to alter traffic-sign pictograms (i.e., symbols within the signs) a
    
[^5]: 教导大型语言模型忘记隐私

    Teach Large Language Models to Forget Privacy. (arXiv:2401.00870v1 [cs.CR])

    [http://arxiv.org/abs/2401.00870](http://arxiv.org/abs/2401.00870)

    这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。

    

    大型语言模型（LLM）已被证明具有强大的能力，但隐私泄露的风险仍然是一个重要问题。传统的保护隐私方法，如差分隐私和同态加密，在只有黑盒API的环境下是不足够的，要求模型透明性或大量计算资源。我们提出了Prompt2Forget（P2F），这是第一个设计用于解决LLM本地隐私挑战的框架，通过教导LLM忘记来实现。该方法涉及将完整问题分解为较小的片段，生成虚构的答案，并使模型对原始输入的记忆模糊化。我们根据不同领域的包含隐私敏感信息的问题创建了基准数据集。P2F实现了零-shot泛化，可以在多种应用场景下自适应，无需手动调整。实验结果表明，P2F具有很强的模糊化LLM记忆的能力，而不会损失任何实用性。

    Large Language Models (LLMs) have proven powerful, but the risk of privacy leakage remains a significant concern. Traditional privacy-preserving methods, such as Differential Privacy and Homomorphic Encryption, are inadequate for black-box API-only settings, demanding either model transparency or heavy computational resources. We propose Prompt2Forget (P2F), the first framework designed to tackle the LLM local privacy challenge by teaching LLM to forget. The method involves decomposing full questions into smaller segments, generating fabricated answers, and obfuscating the model's memory of the original input. A benchmark dataset was crafted with questions containing privacy-sensitive information from diverse fields. P2F achieves zero-shot generalization, allowing adaptability across a wide range of use cases without manual adjustments. Experimental results indicate P2F's robust capability to obfuscate LLM's memory, attaining a forgetfulness score of around 90\% without any utility los
    
[^6]: Nebula:用于动态恶意软件分析的自注意力网络

    Nebula: Self-Attention for Dynamic Malware Analysis. (arXiv:2310.10664v1 [cs.CR])

    [http://arxiv.org/abs/2310.10664](http://arxiv.org/abs/2310.10664)

    Nebula是一个自注意力网络，用于动态分析恶意软件。它能够概括不同的行为表示和格式，并结合动态日志报告中的异构信息。实验证明Nebula在三个重要任务上表现出色。

    

    动态分析通过在受控环境中执行程序并将其行为存储在日志报告中，可以检测Windows恶意软件。先前的工作已经开始在这些报告上训练机器学习模型以进行恶意软件检测或分类。然而，大多数方法仅考虑了卷积和长短期记忆网络，只关注运行时调用的API，并未考虑其他相关的异构信息来源，如网络和文件操作。此外，代码和预训练模型很难获取，这限制了该研究领域中结果的可重现性。在本文中，我们通过提出Nebula来克服这些限制，这是一个多功能的、基于自注意力的转换器神经架构，可以概括不同的行为表示和格式，结合动态日志报告中的异构信息。我们展示了Nebula的有效性，它在三个重要任务上的实验中表现出色。

    Dynamic analysis enables detecting Windows malware by executing programs in a controlled environment, and storing their actions in log reports. Previous work has started training machine learning models on such reports to perform either malware detection or malware classification. However, most of the approaches (i) have only considered convolutional and long-short term memory networks, (ii) they have been built focusing only on APIs called at runtime, without considering other relevant though heterogeneous sources of information like network and file operations, and (iii) the code and pretrained models are hardly available, hindering reproducibility of results in this research area. In this work, we overcome these limitations by presenting Nebula, a versatile, self-attention transformer-based neural architecture that can generalize across different behavior representations and formats, combining heterogeneous information from dynamic log reports. We show the efficacy of Nebula on thre
    

