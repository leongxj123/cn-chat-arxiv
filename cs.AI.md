# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write](https://arxiv.org/abs/2402.05804) | InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。 |
| [^2] | [Obtaining Explainable Classification Models using Distributionally Robust Optimization.](http://arxiv.org/abs/2311.01994) | 本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。 |

# 详细

[^1]: InkSight：通过学习阅读和书写实现离线到在线手写转换

    InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write

    [https://arxiv.org/abs/2402.05804](https://arxiv.org/abs/2402.05804)

    InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。

    

    数字笔记正在变得越来越受欢迎，提供了一种耐用、可编辑和易于索引的存储笔记的方式，即矢量化形式的数字墨水。然而，这种笔记方式与传统的纸笔记方式之间仍存在显著差距，而传统纸笔记方式仍受到绝大多数人的青睐。我们的工作InkSight旨在弥合这种差距，使实体笔记者能够轻松地将他们的作品（离线手写）转换为数字墨水（在线手写），这个过程我们称之为Derendering。之前关于此主题的研究集中在图像的几何属性上，导致了在训练领域之外的有限泛化能力。我们的方法结合了阅读和书写的先验知识，允许在缺乏大量配对样本的情况下训练模型，而这些配对样本很难获取。据我们所知，这是第一个有效地对具有多样化视觉特征和背景的任意照片中的手写文本进行Derendering的工作。

    Digital note-taking is gaining popularity, offering a durable, editable, and easily indexable way of storing notes in the vectorized form, known as digital ink. However, a substantial gap remains between this way of note-taking and traditional pen-and-paper note-taking, a practice still favored by a vast majority. Our work, InkSight, aims to bridge the gap by empowering physical note-takers to effortlessly convert their work (offline handwriting) to digital ink (online handwriting), a process we refer to as Derendering. Prior research on the topic has focused on the geometric properties of images, resulting in limited generalization beyond their training domains. Our approach combines reading and writing priors, allowing training a model in the absence of large amounts of paired samples, which are difficult to obtain. To our knowledge, this is the first work that effectively derenders handwritten text in arbitrary photos with diverse visual characteristics and backgrounds. Furthermore,
    
[^2]: 利用分布鲁棒优化获取可解释的分类模型

    Obtaining Explainable Classification Models using Distributionally Robust Optimization. (arXiv:2311.01994v1 [stat.ML])

    [http://arxiv.org/abs/2311.01994](http://arxiv.org/abs/2311.01994)

    本论文介绍了一种利用分布鲁棒优化获取可解释的分类模型的方法，通过构建稀疏的规则集合来同时解决规则集的稀疏性和预测准确性之间的权衡，从而保证泛化性能并降低计算成本。

    

    对于人类用户来说，模型的可解释性对于理解提议分类器如何根据特征值给数据分配标签至关重要。我们研究使用特征值规则集构建的广义线性模型，该模型可以捕捉非线性依赖和交互作用。规则集的稀疏性和预测准确性之间存在固有的权衡。使用现有方法来找到合适的稀疏度选择（例如通过交叉验证）计算成本很高。我们提出了一种新的公式来学习同时解决这些竞争因素的规则集合。通过利用分布鲁棒优化来确保良好的泛化性能，同时保持低计算成本。该公式利用列生成有效地搜索规则集合的空间并构建稀疏的规则集合，与随机森林或Boosting及其变体等技术相比。我们提出了理论结果来推动这一公式的发展。

    Model explainability is crucial for human users to be able to interpret how a proposed classifier assigns labels to data based on its feature values. We study generalized linear models constructed using sets of feature value rules, which can capture nonlinear dependencies and interactions. An inherent trade-off exists between rule set sparsity and its prediction accuracy. It is computationally expensive to find the right choice of sparsity -- e.g., via cross-validation -- with existing methods. We propose a new formulation to learn an ensemble of rule sets that simultaneously addresses these competing factors. Good generalization is ensured while keeping computational costs low by utilizing distributionally robust optimization. The formulation utilizes column generation to efficiently search the space of rule sets and constructs a sparse ensemble of rule sets, in contrast with techniques like random forests or boosting and their variants. We present theoretical results that motivate an
    

