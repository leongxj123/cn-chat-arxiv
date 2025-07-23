# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment.](http://arxiv.org/abs/2311.01059) | 本研究提出了一种名为ROAM的方法，通过利用先前学习到的行为来实时调节机器人在部署过程中应对未曾见过的情况。在测试中，ROAM可以在单个阶段内实现快速适应，并且在模拟环境和真实场景中取得了成功，具有较高的效率和适应性。 |

# 详细

[^1]: 在部署时进行实时调节：用于单机器人部署的行为调控

    Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment. (arXiv:2311.01059v1 [cs.RO])

    [http://arxiv.org/abs/2311.01059](http://arxiv.org/abs/2311.01059)

    本研究提出了一种名为ROAM的方法，通过利用先前学习到的行为来实时调节机器人在部署过程中应对未曾见过的情况。在测试中，ROAM可以在单个阶段内实现快速适应，并且在模拟环境和真实场景中取得了成功，具有较高的效率和适应性。

    

    为了在现实世界中取得成功，机器人必须应对训练过程中未曾见过的情况。本研究探讨了在部署过程中针对这些新场景的实时调节问题，通过利用先前学习到的多样化行为库。我们的方法，RObust Autonomous Modulation（ROAM），引入了基于预训练行为的感知价值的机制，以在特定情况下选择和调整预训练行为。关键是，这种调节过程在测试时的单个阶段内完成，无需任何人类监督。我们对选择机制进行了理论分析，并证明了ROAM使得机器人能够在模拟环境和真实的四足动物Go1上快速适应动态变化，甚至在脚上套着滚轮滑鞋的情况下成功前进。与现有方法相比，我们的方法在面对各种分布情况的部署时能够以超过2倍的效率进行调节，通过有效选择来实现适应。

    To succeed in the real world, robots must cope with situations that differ from those seen during training. We study the problem of adapting on-the-fly to such novel scenarios during deployment, by drawing upon a diverse repertoire of previously learned behaviors. Our approach, RObust Autonomous Modulation (ROAM), introduces a mechanism based on the perceived value of pre-trained behaviors to select and adapt pre-trained behaviors to the situation at hand. Crucially, this adaptation process all happens within a single episode at test time, without any human supervision. We provide theoretical analysis of our selection mechanism and demonstrate that ROAM enables a robot to adapt rapidly to changes in dynamics both in simulation and on a real Go1 quadruped, even successfully moving forward with roller skates on its feet. Our approach adapts over 2x as efficiently compared to existing methods when facing a variety of out-of-distribution situations during deployment by effectively choosing
    

