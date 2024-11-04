# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simulator-Free Visual Domain Randomization via Video Games](https://rss.arxiv.org/abs/2402.01335) | 本研究提出了一种名为BehAVE的视频理解框架，借助现有的商业视频游戏实现领域随机化，无需仿真器的支持。通过利用游戏中丰富的视觉多样性进行随机化，以及通过玩家行为的文本描述来指导具有相似内容的视频的对齐，BehAVE在领域随机化方面展现了鲁棒性。 |
| [^2] | [Make Continual Learning Stronger via C-Flat](https://arxiv.org/abs/2404.00986) | 通过C-Flat方法，我们提出了一种更平坦的损失景观，可用于持续学习，简化了模型训练过程并提高了模型泛化能力。 |
| [^3] | [A Critical Look at Classic Test-Time Adaptation Methods in Semantic Segmentation.](http://arxiv.org/abs/2310.05341) | 这项研究对语义分割中的经典测试时适应方法进行了批判性探究，揭示了分割TTA所面临的独特挑战，并发现经典TTA策略在这一任务中并不有效。 |

# 详细

[^1]: 通过视频游戏实现无仿真器视觉领域随机化

    Simulator-Free Visual Domain Randomization via Video Games

    [https://rss.arxiv.org/abs/2402.01335](https://rss.arxiv.org/abs/2402.01335)

    本研究提出了一种名为BehAVE的视频理解框架，借助现有的商业视频游戏实现领域随机化，无需仿真器的支持。通过利用游戏中丰富的视觉多样性进行随机化，以及通过玩家行为的文本描述来指导具有相似内容的视频的对齐，BehAVE在领域随机化方面展现了鲁棒性。

    

    领域随机化是一种有效的计算机视觉技术，用于提高视觉模型在视觉上截然不同但内容相似的领域中的传递性。然而，现有的方法大量依赖于调整复杂和专门的仿真引擎，这些引擎的构建很困难，进而影响了它们的可行性和可扩展性。本文介绍了BehAVE，一种视频理解框架，它独特地利用现有的商业视频游戏来实现领域随机化，而无需访问它们的仿真引擎。在BehAVE下，(1) 视频游戏固有的丰富视觉多样性成为随机化的来源，(2) 玩家行为 - 通过动作的文本描述进行语义表示 - 引导具有相似内容的视频的对齐。我们在各种视频和文本基础模型上测试了BehAVE，并报告了它在领域随机化方面的鲁棒性。

    Domain randomization is an effective computer vision technique for improving transferability of vision models across visually distinct domains exhibiting similar content. Existing approaches, however, rely extensively on tweaking complex and specialized simulation engines that are difficult to construct, subsequently affecting their feasibility and scalability. This paper introduces BehAVE, a video understanding framework that uniquely leverages the plethora of existing commercial video games for domain randomization, without requiring access to their simulation engines. Under BehAVE (1) the inherent rich visual diversity of video games acts as the source of randomization and (2) player behavior -- represented semantically via textual descriptions of actions -- guides the *alignment* of videos with similar content. We test BehAVE on 25 games of the first-person shooter (FPS) genre across various video and text foundation models and we report its robustness for domain randomization. Beh
    
[^2]: 通过C-Flat使持续学习更强大

    Make Continual Learning Stronger via C-Flat

    [https://arxiv.org/abs/2404.00986](https://arxiv.org/abs/2404.00986)

    通过C-Flat方法，我们提出了一种更平坦的损失景观，可用于持续学习，简化了模型训练过程并提高了模型泛化能力。

    

    持续学习中模型的泛化能力对于处理连续到达任务的动态更新知识是至关重要的，为了解决持续学习中的敏感性-稳定性困境。研究证明，通过最小化权重损失景观的陡峭度，寻找位于具有统一低损失或平稳梯度的邻域中的平坦最小值，是一种强大的训练方式，相较于基于损失最小化的优化器如SGD来提高模型的泛化性。然而，只有少数作品讨论了这种训练方式在持续学习中的应用，证明特定设计的零阶陡峭度优化器可以提升持续学习性能。在这项工作中，我们提出了一种名为Continual Flatness（C-Flat）的方法，具有为持续学习定制的更平坦的损失景观。C-Flat只需一行代码即可轻松调用，并可与任何持续学习方法插播。C-Flat应用于所有持续学习类别的一般框架，并与损失最小化优化器进行了彻底比较。

    arXiv:2404.00986v1 Announce Type: new  Abstract: Model generalization ability upon incrementally acquiring dynamically updating knowledge from sequentially arriving tasks is crucial to tackle the sensitivity-stability dilemma in Continual Learning (CL). Weight loss landscape sharpness minimization seeking for flat minima lying in neighborhoods with uniform low loss or smooth gradient is proven to be a strong training regime improving model generalization compared with loss minimization based optimizer like SGD. Yet only a few works have discussed this training regime for CL, proving that dedicated designed zeroth-order sharpness optimizer can improve CL performance. In this work, we propose a Continual Flatness (C-Flat) method featuring a flatter loss landscape tailored for CL. C-Flat could be easily called with only one line of code and is plug-and-play to any CL methods. A general framework of C-Flat applied to all CL categories and a thorough comparison with loss minima optimizer an
    
[^3]: 对语义分割中经典的测试时适应方法的批判性探究

    A Critical Look at Classic Test-Time Adaptation Methods in Semantic Segmentation. (arXiv:2310.05341v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.05341](http://arxiv.org/abs/2310.05341)

    这项研究对语义分割中的经典测试时适应方法进行了批判性探究，揭示了分割TTA所面临的独特挑战，并发现经典TTA策略在这一任务中并不有效。

    

    测试时适应（TTA）旨在将最初在训练数据上训练的模型适应于测试数据中的可能分布变化。然而，大多数现有的TTA研究都集中在分类任务上，对于语义分割的TTA探索非常有限。这种对分类的突出重视可能导致许多新手和工程师错误地认为为分类设计的经典TTA方法可以直接应用于分割任务。然而，这一假设仍未经验证，是一个待解决的问题。为了解决这个问题，我们进行了一项系统的实证研究，揭示了分割TTA的独特挑战，并确定经典TTA策略是否可以有效应对这一任务。我们全面的结果得出了三个关键观察结果。首先，常用于分类TTA的经典批归一化更新策略只能带来轻微的性能改善，在某些情况下甚至会对结果产生逆向影响。

    Test-time adaptation (TTA) aims to adapt a model, initially trained on training data, to potential distribution shifts in the test data. Most existing TTA studies, however, focus on classification tasks, leaving a notable gap in the exploration of TTA for semantic segmentation. This pronounced emphasis on classification might lead numerous newcomers and engineers to mistakenly assume that classic TTA methods designed for classification can be directly applied to segmentation. Nonetheless, this assumption remains unverified, posing an open question. To address this, we conduct a systematic, empirical study to disclose the unique challenges of segmentation TTA, and to determine whether classic TTA strategies can effectively address this task. Our comprehensive results have led to three key observations. First, the classic batch norm updating strategy, commonly used in classification TTA, only brings slight performance improvement, and in some cases it might even adversely affect the resu
    

