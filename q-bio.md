# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Language Models Are Versatile Protein Learners](https://arxiv.org/abs/2402.18567) | DPLM是一种多才多艺的蛋白质语言模型，通过扩散生成式预训练使其具有更好的蛋白质理解能力，并展示了在生成和预测任务中的优越表现。 |
| [^2] | [COMPASS: Computational Mapping of Patient-Therapist Alliance Strategies with Language Modeling](https://arxiv.org/abs/2402.14701) | 本文提出了一种名为COMPASS的新框架，通过分析心理治疗会话中的自然语言，直接推断治疗工作联盟，为临床精神病学提供了可解释性，并在识别与正在治疗的疾病相关的新兴模式方面发挥作用。 |
| [^3] | [Contrastive losses as generalized models of global epistasis.](http://arxiv.org/abs/2305.03136) | 对比损失函数可以用于提取全局上位联系模型所隐含的稀疏潜在函数，这种方法可能为蛋白质工程和相关领域的适应性函数推断提供有用的通用框架。 |

# 详细

[^1]: 扩散语言模型是多才多艺的蛋白质学习者

    Diffusion Language Models Are Versatile Protein Learners

    [https://arxiv.org/abs/2402.18567](https://arxiv.org/abs/2402.18567)

    DPLM是一种多才多艺的蛋白质语言模型，通过扩散生成式预训练使其具有更好的蛋白质理解能力，并展示了在生成和预测任务中的优越表现。

    

    本文提出了扩散蛋白质语言模型（DPLM），这是一种多才多艺的蛋白质语言模型，展示了对蛋白质序列具有强大的生成和预测能力。我们首先在一种生成式自监督离散扩散概率框架中从进化规模的蛋白质序列中预训练可扩展的DPLM，这为蛋白质的语言建模提供了基本方法。在预训练之后，DPLM展示了生成出符合结构的、新颖的、多样的蛋白质序列的能力。我们进一步展示了所提出的扩散生成式预训练使得DPLM对蛋白质具有更好的理解，使其成为一种更优秀的表示学习者，可以为各种预测任务进行微调，并且与ESM2（Lin et al., 2022）相比表现优异。此外，DPLM可以针对各种需求进行定制，展示了其在多种情况下进行条件生成的实力。

    arXiv:2402.18567v1 Announce Type: new  Abstract: This paper introduces diffusion protein language model (DPLM), a versatile protein language model that demonstrates strong generative and predictive capabilities for protein sequences. We first pre-train scalable DPLMs from evolutionary-scale protein sequences within a generative self-supervised discrete diffusion probabilistic framework, which generalizes language modeling for proteins in a principled way. After pre-training, DPLM exhibits the ability to generate structurally plausible, novel, and diverse protein sequences for unconditional generation. We further demonstrate the proposed diffusion generative pre-training makes DPLM possess a better understanding of proteins, making it a superior representation learner, which can be fine-tuned for various predictive tasks, comparing favorably to ESM2 (Lin et al., 2022). Moreover, DPLM can be tailored for various needs, which showcases its prowess of conditional generation in several ways
    
[^2]: COMPASS：利用语言建模对患者-治疗师联盟策略进行计算映射

    COMPASS: Computational Mapping of Patient-Therapist Alliance Strategies with Language Modeling

    [https://arxiv.org/abs/2402.14701](https://arxiv.org/abs/2402.14701)

    本文提出了一种名为COMPASS的新框架，通过分析心理治疗会话中的自然语言，直接推断治疗工作联盟，为临床精神病学提供了可解释性，并在识别与正在治疗的疾病相关的新兴模式方面发挥作用。

    

    治疗工作联盟是预测心理治疗治疗成功的关键因素。传统上，工作联盟评估依赖于治疗师和患者填写的问卷。本文提出了COMPASS，一个新颖的框架，可直接从心理治疗课程中使用的自然语言中推断治疗工作联盟。我们的方法利用先进的大型语言模型分析心理治疗会话的转录，并将其与工作联盟清单中陈述的分布式表示进行比较。通过分析涵盖多种精神疾病的超过950个会话的数据集，我们展示了我们的方法在显微地映射患者-治疗师对齐轨迹方面的有效性，并为临床精神病学提供解释性，并在识别与正在治疗的疾病相关的新兴模式方面提供可解释性。通过使用各种神经主题模式

    arXiv:2402.14701v1 Announce Type: cross  Abstract: The therapeutic working alliance is a critical factor in predicting the success of psychotherapy treatment. Traditionally, working alliance assessment relies on questionnaires completed by both therapists and patients. In this paper, we present COMPASS, a novel framework to directly infer the therapeutic working alliance from the natural language used in psychotherapy sessions. Our approach utilizes advanced large language models to analyze transcripts of psychotherapy sessions and compare them with distributed representations of statements in the working alliance inventory. Analyzing a dataset of over 950 sessions covering diverse psychiatric conditions, we demonstrate the effectiveness of our method in microscopically mapping patient-therapist alignment trajectories and providing interpretability for clinical psychiatry and in identifying emerging patterns related to the condition being treated. By employing various neural topic mode
    
[^3]: 对比损失作为全局上位联系模型的广义模型

    Contrastive losses as generalized models of global epistasis. (arXiv:2305.03136v1 [q-bio.PE])

    [http://arxiv.org/abs/2305.03136](http://arxiv.org/abs/2305.03136)

    对比损失函数可以用于提取全局上位联系模型所隐含的稀疏潜在函数，这种方法可能为蛋白质工程和相关领域的适应性函数推断提供有用的通用框架。

    

    适应性函数将生物序列的大组合空间映射到所关注的特性上。从实验数据中推断这些多模态函数是现代蛋白质工程中的核心任务。全局上位联系模型是一类有效且有物理基础的模型，可用于估计从观察数据中推断适应性函数。这些模型假设稀疏的潜在函数通过单调非线性变换以发射可测的适应性。在这里，我们展示了最小化对比损失函数（如 Bradley-Terry 损失）是提取全局上位联系所隐示的稀疏潜在函数的一种简单灵活的技术。我们通过适应性-上位联系不确定性原理争辩，全局上位联系模型中的非线性可以产生不具备稀疏表示的观察适应性函数，因此可能不适合使用均方误差（MSE）损失（一种常见的做法）从观察中学习。我们表明，对比损失可用于推断不适合 MSE 损失的适应性函数，并且全局上位联系模型可以解释为一种规则化的对比损失模型。我们的结果表明，这种方法可能为蛋白质工程和相关领域的适应性函数推断提供有用的通用框架。

    Fitness functions map large combinatorial spaces of biological sequences to properties of interest. Inferring these multimodal functions from experimental data is a central task in modern protein engineering. Global epistasis models are an effective and physically-grounded class of models for estimating fitness functions from observed data. These models assume that a sparse latent function is transformed by a monotonic nonlinearity to emit measurable fitness. Here we demonstrate that minimizing contrastive loss functions, such as the Bradley-Terry loss, is a simple and flexible technique for extracting the sparse latent function implied by global epistasis. We argue by way of a fitness-epistasis uncertainty principle that the nonlinearities in global epistasis models can produce observed fitness functions that do not admit sparse representations, and thus may be inefficient to learn from observations when using a Mean Squared Error (MSE) loss (a common practice). We show that contrasti
    

