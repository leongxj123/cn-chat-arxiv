# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretable Semiotics Networks Representing Awareness](https://arxiv.org/abs/2310.05212) | 这个研究描述了一个计算模型，通过追踪和模拟物体感知以及其在交流中所传达的表示来模拟人类的意识。相比于大多数无法解释的神经网络，该模型具有解释性，并可以通过构建新网络来定义物体感知。 |
| [^2] | [STANCE-C3: Domain-adaptive Cross-target Stance Detection via Contrastive Learning and Counterfactual Generation.](http://arxiv.org/abs/2309.15176) | STANCE-C3是一种通过对比学习和反事实生成进行领域自适应的跨目标立场检测模型，用于推断人们对于普遍或有争议话题的观点。在解决数据分布偏移和缺乏领域特定标注数据的挑战上具有重要贡献。 |

# 详细

[^1]: 可解释的符号网络代表意识的知觉

    Interpretable Semiotics Networks Representing Awareness

    [https://arxiv.org/abs/2310.05212](https://arxiv.org/abs/2310.05212)

    这个研究描述了一个计算模型，通过追踪和模拟物体感知以及其在交流中所传达的表示来模拟人类的意识。相比于大多数无法解释的神经网络，该模型具有解释性，并可以通过构建新网络来定义物体感知。

    

    人类每天都感知物体，并通过各种渠道传达他们的感知。在这里，我们描述了一个计算模型，追踪和模拟物体的感知以及它们在交流中所传达的表示。我们描述了我们内部表示的两个关键组成部分（"观察到的"和"看到的"），并将它们与熟悉的计算机视觉概念（编码和解码）相关联。这些元素被合并在一起形成符号网络，模拟了物体感知和人类交流中的意识。如今，大多数神经网络都是不可解释的。另一方面，我们的模型克服了这个限制。实验证明了该模型的可见性。我们人的物体感知模型使我们能够通过网络定义物体感知。我们通过构建一个包括基准分类器和额外层的新网络来演示这一点。这个层产生了图像的感知。

    Humans perceive objects daily and communicate their perceptions using various channels. Here, we describe a computational model that tracks and simulates objects' perception and their representations as they are conveyed in communication.   We describe two key components of our internal representation ("observed" and "seen") and relate them to familiar computer vision notions (encoding and decoding). These elements are joined together to form semiotics networks, which simulate awareness in object perception and human communication.   Nowadays, most neural networks are uninterpretable. On the other hand, our model overcomes this limitation. The experiments demonstrates the visibility of the model.   Our model of object perception by a person allows us to define object perception by a network. We demonstrate this with an example of an image baseline classifier by constructing a new network that includes the baseline classifier and an additional layer. This layer produces the images "perc
    
[^2]: STANCE-C3: 通过对比学习和反事实生成进行领域自适应的跨目标立场检测

    STANCE-C3: Domain-adaptive Cross-target Stance Detection via Contrastive Learning and Counterfactual Generation. (arXiv:2309.15176v1 [cs.CL])

    [http://arxiv.org/abs/2309.15176](http://arxiv.org/abs/2309.15176)

    STANCE-C3是一种通过对比学习和反事实生成进行领域自适应的跨目标立场检测模型，用于推断人们对于普遍或有争议话题的观点。在解决数据分布偏移和缺乏领域特定标注数据的挑战上具有重要贡献。

    

    立场检测是通过推断一个人在特定问题上的立场或观点，以推断对于普遍或有争议的话题的普遍看法，例如COVID-19疫情期间的健康政策。现有的立场检测模型在训练时往往在单个领域（例如COVID-19）和特定目标话题（例如口罩规定）上表现良好，但在其他领域或目标中往往表现不佳，这是由于数据的分布偏移。然而，构建高性能的领域特定立场检测模型需要大量与目标领域相关的已标注数据，但这样的数据集往往不容易获取。这就面临着一个挑战，因为标注数据的过程代价高昂且耗时。为了应对这些挑战，我们提出了一种新颖的立场检测模型，称为通过对比学习和反事实生成进行领域自适应的跨目标立场检测（STANCE-C3）。

    Stance detection is the process of inferring a person's position or standpoint on a specific issue to deduce prevailing perceptions toward topics of general or controversial interest, such as health policies during the COVID-19 pandemic. Existing models for stance detection are trained to perform well for a single domain (e.g., COVID-19) and a specific target topic (e.g., masking protocols), but are generally ineffectual in other domains or targets due to distributional shifts in the data. However, constructing high-performing, domain-specific stance detection models requires an extensive corpus of labeled data relevant to the targeted domain, yet such datasets are not readily available. This poses a challenge as the process of annotating data is costly and time-consuming. To address these challenges, we introduce a novel stance detection model coined domain-adaptive Cross-target STANCE detection via Contrastive learning and Counterfactual generation (STANCE-C3) that uses counterfactua
    

