# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What's in Your "Safe" Data?: Identifying Benign Data that Breaks Safety](https://arxiv.org/abs/2404.01099) | 通过双向锚定方法，识别那些在微调后更可能降低模型安全性的良性数据子集，提高模型对有害请求的响应率。 |
| [^2] | [How to Make the Gradients Small Privately: Improved Rates for Differentially Private Non-Convex Optimization](https://arxiv.org/abs/2402.11173) | 提出了一种设计具有差分隐私算法的简单灵活框架，用于寻找非凸损失函数的近似稳定点，并获得了改进和有时是最优的速率。 |
| [^3] | [Disparate Impact on Group Accuracy of Linearization for Private Inference](https://arxiv.org/abs/2402.03629) | 本文研究了线性化对隐私推断中群体准确性的影响，发现减少ReLU激活函数数量会不成比例地降低少数群体的准确性，而对于多数群体则几乎没有影响。采用简单的微调步骤可以解决这个问题。 |
| [^4] | [PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety](https://arxiv.org/abs/2401.11880) | PsySafe提出了一个综合框架，通过深入探讨智能体心理学，揭示智能体的黑暗心理状态对安全构成威胁，并提出了有效的风险缓解策略。 |
| [^5] | [Malla: Demystifying Real-world Large Language Model Integrated Malicious Services.](http://arxiv.org/abs/2401.03315) | 本研究对212个真实的恶意服务（Malla）进行了系统研究，揭示了它们在地下市场的扩散和对公共LLM服务的影响，以及其使用的策略和技术。 |
| [^6] | [DPM: Clustering Sensitive Data through Separation.](http://arxiv.org/abs/2307.02969) | 本文提出了差分隐私聚类算法DPM，通过搜索准确的数据点分离器来进行隐私保护的聚类。关键贡献是识别大间隔分离器并合理分配隐私预算。 |

# 详细

[^1]: 你的“安全”数据中有什么？：识别破坏安全性的良性数据

    What's in Your "Safe" Data?: Identifying Benign Data that Breaks Safety

    [https://arxiv.org/abs/2404.01099](https://arxiv.org/abs/2404.01099)

    通过双向锚定方法，识别那些在微调后更可能降低模型安全性的良性数据子集，提高模型对有害请求的响应率。

    

    当前的大型语言模型（LLMs），即使经过调整以确保安全性和对齐性，也容易被越狱。一些研究表明，只是进一步使用良性数据（即没有有害内容的数据）对一个对齐模型进行微调，会导致安全性大幅下降。我们深入探讨良性微调不经意间导致越狱的数据中心方面。首先，我们通过两种视角表征微调数据：表示和梯度空间。此外，我们提出了一种双向锚定方法，该方法优先考虑靠近有害示例并远离良性示例的数据点。通过这样做，我们的方法有效地识别出更有可能在微调后降低模型安全性的良性数据子集。仅仅训练100个这些看似良性的数据点，就可以使微调模型肯定地回应超过70％的被测试的有害请求，相比之下，...

    arXiv:2404.01099v1 Announce Type: cross  Abstract: Current Large Language Models (LLMs), even those tuned for safety and alignment, are susceptible to jailbreaking. Some have found that just further fine-tuning an aligned model with benign data (i.e., data without harmful content) surprisingly leads to substantial degradation in safety. We delve into the data-centric aspects of why benign fine-tuning inadvertently contributes to jailbreaking. First, we represent fine-tuning data through two lenses: representation and gradient spaces. Furthermore, we propose a bi-directional anchoring method that prioritizes data points that are close to harmful examples and distant from benign ones. By doing so, our approach effectively identifies subsets of benign data that are more likely to degrade the model's safety after fine-tuning. Training on just 100 of these seemingly benign datapoints can lead to the fine-tuned model affirmatively responding to > 70% of tested harmful requests, compared to <
    
[^2]: 如何在隐私条件下使梯度变得更小：改进的差分隐私非凸优化速率

    How to Make the Gradients Small Privately: Improved Rates for Differentially Private Non-Convex Optimization

    [https://arxiv.org/abs/2402.11173](https://arxiv.org/abs/2402.11173)

    提出了一种设计具有差分隐私算法的简单灵活框架，用于寻找非凸损失函数的近似稳定点，并获得了改进和有时是最优的速率。

    

    我们提供了一个简单灵活的框架，用于设计具有差分隐私算法，以找到非凸损失函数的近似稳定点。我们的框架基于使用私有的近似风险最小化器来“热启动”另一个用于寻找稳定点的私有算法。我们利用这个框架来获得对几类非凸损失函数的改进甚至是最优速率。首先，我们改进了寻找平滑非凸经验损失函数稳定点的速率。其次，我们专门针对夸萨-凸函数，这种函数概括了星-凸函数，并在学习动态系统和训练一些神经网络时出现。我们为这个类别实现了最优速率。第三，我们提供了一种对满足Kurdyka-Lojasiewicz（KL）条件的函数寻找稳定点的最优算法。例如，超参数化神经网络经常满足这个条件。

    arXiv:2402.11173v1 Announce Type: new  Abstract: We provide a simple and flexible framework for designing differentially private algorithms to find approximate stationary points of non-convex loss functions. Our framework is based on using a private approximate risk minimizer to "warm start" another private algorithm for finding stationary points. We use this framework to obtain improved, and sometimes optimal, rates for several classes of non-convex loss functions. First, we obtain improved rates for finding stationary points of smooth non-convex empirical loss functions. Second, we specialize to quasar-convex functions, which generalize star-convex functions and arise in learning dynamical systems and training some neural nets. We achieve the optimal rate for this class. Third, we give an optimal algorithm for finding stationary points of functions satisfying the Kurdyka-Lojasiewicz (KL) condition. For example, over-parameterized neural networks often satisfy this condition. Fourth, 
    
[^3]: 私有推断的线性化对群体准确性的不对称影响

    Disparate Impact on Group Accuracy of Linearization for Private Inference

    [https://arxiv.org/abs/2402.03629](https://arxiv.org/abs/2402.03629)

    本文研究了线性化对隐私推断中群体准确性的影响，发现减少ReLU激活函数数量会不成比例地降低少数群体的准确性，而对于多数群体则几乎没有影响。采用简单的微调步骤可以解决这个问题。

    

    确保对具有密码安全性的数据进行隐私保护的推断是一个众所周知的计算挑战。为了减轻非线性激活函数中昂贵的加密计算的瓶颈，最近的方法建议在线性神经网络中线性化目标部分的激活函数。这种技术可以显著减少运行时间，对准确性的影响往往可以忽略不计。在本文中，我们证明了这种计算优势可能导致公平性成本增加。具体而言，我们发现减少ReLU激活函数数量会不成比例地降低少数群体的准确性，而对于多数群体则几乎没有影响。为了解释这些观察结果，我们在对决策边界性质进行限制性假设的基础上提供了数学解释，同时还展示了这个问题在广泛使用的数据集和体系结构中的普遍性。最后，我们展示了如何通过简单的程序改变线性模型的微调步骤来解决这个问题。

    Ensuring privacy-preserving inference on cryptographically secure data is a well-known computational challenge. To alleviate the bottleneck of costly cryptographic computations in non-linear activations, recent methods have suggested linearizing a targeted portion of these activations in neural networks. This technique results in significantly reduced runtimes with often negligible impacts on accuracy. In this paper, we demonstrate that such computational benefits may lead to increased fairness costs. Specifically, we find that reducing the number of ReLU activations disproportionately decreases the accuracy for minority groups compared to majority groups. To explain these observations, we provide a mathematical interpretation under restricted assumptions about the nature of the decision boundary, while also showing the prevalence of this problem across widely used datasets and architectures. Finally, we show how a simple procedure altering the fine-tuning step for linearized models ca
    
[^4]: PsySafe：基于心理学的多智能体系统安全攻击、防御和评估的综合框架

    PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety

    [https://arxiv.org/abs/2401.11880](https://arxiv.org/abs/2401.11880)

    PsySafe提出了一个综合框架，通过深入探讨智能体心理学，揭示智能体的黑暗心理状态对安全构成威胁，并提出了有效的风险缓解策略。

    

    多智能体系统在加入大型语言模型（LLMs）后，展现出了集体智能的深远能力。然而，这种智能被恶意使用可能带来重大风险。迄今为止，关于多智能体系统安全问题的全面研究仍然有限。本文通过创新的视角探索了这些问题，发现智能体的黑暗心理状态构成了对安全的重大威胁。为了解决这些问题，我们提出了一个以智能体心理学为基础的综合框架（PsySafe），关注三个关键领域：首先，识别智能体中的黑暗人格特征如何导致风险行为；其次，从心理和行为角度评估多智能体系统的安全性；第三，制定有效的策略来减轻这些风险。我们的实验揭示

    arXiv:2401.11880v2 Announce Type: replace-cross  Abstract: Multi-agent systems, when enhanced with Large Language Models (LLMs), exhibit profound capabilities in collective intelligence. However, the potential misuse of this intelligence for malicious purposes presents significant risks. To date, comprehensive research on the safety issues associated with multi-agent systems remains limited. In this paper, we explore these concerns through the innovative lens of agent psychology, revealing that the dark psychological states of agents constitute a significant threat to safety. To tackle these concerns, we propose a comprehensive framework (PsySafe) grounded in agent psychology, focusing on three key areas: firstly, identifying how dark personality traits in agents can lead to risky behaviors; secondly, evaluating the safety of multi-agent systems from the psychological and behavioral perspectives, and thirdly, devising effective strategies to mitigate these risks. Our experiments reveal
    
[^5]: Malla: 揭秘现实世界中大规模语言模型整合恶意服务

    Malla: Demystifying Real-world Large Language Model Integrated Malicious Services. (arXiv:2401.03315v1 [cs.CR])

    [http://arxiv.org/abs/2401.03315](http://arxiv.org/abs/2401.03315)

    本研究对212个真实的恶意服务（Malla）进行了系统研究，揭示了它们在地下市场的扩散和对公共LLM服务的影响，以及其使用的策略和技术。

    

    大规模语言模型（LLMs）的地下利用，也称为Malla，正在增加，加剧了网络安全威胁，并对LLMs技术的可信度提出了疑问。然而，迄今为止，很少有工作努力去了解这种新型网络犯罪的规模、影响和技术。本文是第一次对212个真实的Malla进行系统研究，揭示了它们在地下市场的扩散，并揭示了它们的操作模式。我们的研究揭开了Malla生态系统，揭示了其显著的增长对当今公共LLM服务的影响。通过对212个Mallas进行研究，我们发现了8个后端LLMs，以及182个绕过公共LLM API保护措施的提示。我们进一步揭示了Mallas使用的策略，包括滥用未经审查的LLMs和通过越狱提示利用公共LLM API。我们的发现有助于更好地理解Malla犯罪行为的实质。

    The underground exploitation of large language models (LLMs) for malicious services (i.e., Malla) is witnessing an uptick, amplifying the cyber threat landscape and posing questions about the trustworthiness of LLM technologies. However, there has been little effort to understand this new cybercrime, in terms of its magnitude, impact, and techniques. In this paper, we conduct the first systematic study on 212 real-world Mallas, uncovering their proliferation in underground marketplaces and exposing their operational modalities. Our study discloses the Malla ecosystem, revealing its significant growth and impact on today's public LLM services. Through examining 212 Mallas, we uncovered eight backend LLMs used by Mallas, along with 182 prompts that circumvent the protective measures of public LLM APIs. We further demystify the tactics employed by Mallas, including the abuse of uncensored LLMs and the exploitation of public LLM APIs through jailbreak prompts. Our findings enable a better 
    
[^6]: DPM: 通过分离聚类敏感数据

    DPM: Clustering Sensitive Data through Separation. (arXiv:2307.02969v1 [cs.CR])

    [http://arxiv.org/abs/2307.02969](http://arxiv.org/abs/2307.02969)

    本文提出了差分隐私聚类算法DPM，通过搜索准确的数据点分离器来进行隐私保护的聚类。关键贡献是识别大间隔分离器并合理分配隐私预算。

    

    隐私保护聚类以无监督方式对数据点进行分组，同时确保敏感信息得以保护。先前的隐私保护聚类关注点在于识别点云的聚集。本文则采取另一种方法，关注于识别适当的分离器以分离数据集。我们引入了新颖的差分隐私聚类算法DPM，以差分隐私的方式搜索准确的数据点分离器。DPM解决了寻找准确分离器的两个关键挑战：识别聚类间的大间隔分离器而不是聚类内的小间隔分离器，以及在开销隐私预算时，优先考虑将数据划分为较大子部分的分离器。利用差分隐私指数机制，DPM通过随机选择具有高效用性的聚类分离器：对于数据集D，如果中心的60%分位数中存在宽的低密度分离器，DPM会发现它。

    Privacy-preserving clustering groups data points in an unsupervised manner whilst ensuring that sensitive information remains protected. Previous privacy-preserving clustering focused on identifying concentration of point clouds. In this paper, we take another path and focus on identifying appropriate separators that split a data set. We introduce the novel differentially private clustering algorithm DPM that searches for accurate data point separators in a differentially private manner. DPM addresses two key challenges for finding accurate separators: identifying separators that are large gaps between clusters instead of small gaps within a cluster and, to efficiently spend the privacy budget, prioritising separators that split the data into large subparts. Using the differentially private Exponential Mechanism, DPM randomly chooses cluster separators with provably high utility: For a data set $D$, if there is a wide low-density separator in the central $60\%$ quantile, DPM finds that
    

