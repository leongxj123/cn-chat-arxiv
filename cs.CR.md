# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial Robustness Through Artifact Design](https://arxiv.org/abs/2402.04660) | 该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。 |

# 详细

[^1]: 通过艺术设计提高对抗性鲁棒性

    Adversarial Robustness Through Artifact Design

    [https://arxiv.org/abs/2402.04660](https://arxiv.org/abs/2402.04660)

    该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。

    

    对抗性示例的出现给机器学习带来了挑战。为了阻碍对抗性示例，大多数防御方法都改变了模型的训练方式（如对抗性训练）或推理过程（如随机平滑）。尽管这些方法显著提高了模型的对抗性鲁棒性，但模型仍然极易受到对抗性示例的影响。在某些领域如交通标志识别中，我们发现对象是按照规范来设计（如标志规范）。为了改善对抗性鲁棒性，我们提出了一种新颖的方法。具体来说，我们提供了一种重新定义规范的方法，对现有规范进行微小的更改，以防御对抗性示例。我们将艺术设计问题建模为一个鲁棒优化问题，并提出了基于梯度和贪婪搜索的方法来解决它。我们在交通标志识别领域对我们的方法进行了评估，使其能够改变交通标志中的象形图标（即标志内的符号）。

    Adversarial examples arose as a challenge for machine learning. To hinder them, most defenses alter how models are trained (e.g., adversarial training) or inference is made (e.g., randomized smoothing). Still, while these approaches markedly improve models' adversarial robustness, models remain highly susceptible to adversarial examples. Identifying that, in certain domains such as traffic-sign recognition, objects are implemented per standards specifying how artifacts (e.g., signs) should be designed, we propose a novel approach for improving adversarial robustness. Specifically, we offer a method to redefine standards, making minor changes to existing ones, to defend against adversarial examples. We formulate the problem of artifact design as a robust optimization problem, and propose gradient-based and greedy search methods to solve it. We evaluated our approach in the domain of traffic-sign recognition, allowing it to alter traffic-sign pictograms (i.e., symbols within the signs) a
    

