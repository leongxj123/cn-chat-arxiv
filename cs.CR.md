# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Security Response through Online Learning with Adaptive Conjectures](https://arxiv.org/abs/2402.12499) | 该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。 |
| [^2] | [Defending Our Privacy With Backdoors.](http://arxiv.org/abs/2310.08320) | 本研究提出了一种基于后门攻击的防御方法，通过对模型进行策略性插入后门，对齐敏感短语与中性术语的嵌入，以删除训练数据中的私人信息。实证结果显示该方法的有效性。 |

# 详细

[^1]: 通过自适应猜想的在线学习实现自动化安全响应

    Automated Security Response through Online Learning with Adaptive Conjectures

    [https://arxiv.org/abs/2402.12499](https://arxiv.org/abs/2402.12499)

    该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。

    

    我们研究了针对IT基础设施的自动化安全响应，并将攻击者和防御者之间的互动形式表述为一个部分观测、非平稳博弈。我们放宽了游戏模型正确规定的标准假设，并考虑每个参与者对模型有一个概率性猜想，可能在某种意义上错误规定，即真实模型的概率为0。这种形式允许我们捕捉关于基础设施和参与者意图的不确定性。为了在线学习有效的游戏策略，我们设计了一种新颖的方法，其中一个参与者通过贝叶斯学习迭代地调整其猜想，并通过推演更新其策略。我们证明了猜想会收敛到最佳拟合，并提供了在具有猜测模型的情况下推演实现性能改进的上限。为了刻画游戏的稳定状态，我们提出了Berk-Nash平衡的一个变种。

    arXiv:2402.12499v1 Announce Type: cross  Abstract: We study automated security response for an IT infrastructure and formulate the interaction between an attacker and a defender as a partially observed, non-stationary game. We relax the standard assumption that the game model is correctly specified and consider that each player has a probabilistic conjecture about the model, which may be misspecified in the sense that the true model has probability 0. This formulation allows us to capture uncertainty about the infrastructure and the intents of the players. To learn effective game strategies online, we design a novel method where a player iteratively adapts its conjecture using Bayesian learning and updates its strategy through rollout. We prove that the conjectures converge to best fits, and we provide a bound on the performance improvement that rollout enables with a conjectured model. To characterize the steady state of the game, we propose a variant of the Berk-Nash equilibrium. We 
    
[^2]: 使用后门技术保护我们的隐私

    Defending Our Privacy With Backdoors. (arXiv:2310.08320v1 [cs.LG])

    [http://arxiv.org/abs/2310.08320](http://arxiv.org/abs/2310.08320)

    本研究提出了一种基于后门攻击的防御方法，通过对模型进行策略性插入后门，对齐敏感短语与中性术语的嵌入，以删除训练数据中的私人信息。实证结果显示该方法的有效性。

    

    在使用未经筛选、常常包含敏感信息的网页数据训练大型人工智能模型的情况下，隐私问题成为了一个重要的关注点。其中一个问题是，攻击者可以利用隐私攻击的方法提取出训练数据的信息。然而，如何在不降低模型性能的情况下去除特定信息是一个不容易解决且具有挑战性的问题。我们提出了一个基于后门攻击的简单而有效的防御方法，用于从模型中删除私人信息，如个人姓名，特别是针对文本编码器的。具体而言，通过策略性地插入后门，我们将敏感短语的嵌入与中性术语的嵌入对齐，例如用"a person"代替人名。我们的实证结果通过对零样本分类器使用专门的隐私攻击测试表明了我们基于后门的防御方法的效果。我们的方法提供了一个新的"双重用途"的视角。

    The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspecti
    

