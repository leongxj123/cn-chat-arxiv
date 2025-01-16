# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synapse: Learning Preferential Concepts from Visual Demonstrations](https://arxiv.org/abs/2403.16689) | Synapse是一种神经符号化方法，旨在从有限演示中高效学习偏好概念，通过将偏好表示为神经符号程序并利用视觉解析、大型语言模型和程序合成相结合的方式来学习个人偏好。 |
| [^2] | [Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving](https://arxiv.org/abs/2403.12176) | 自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。 |
| [^3] | [Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation](https://arxiv.org/abs/2403.10700) | 提出了一个新的基准数据集，首次引入了各种类型的指令错误，考虑到潜在的人类原因，以评估连续环境中 VLN 系统的健壮性 |

# 详细

[^1]: Synapse: 从视觉演示中学习优先概念

    Synapse: Learning Preferential Concepts from Visual Demonstrations

    [https://arxiv.org/abs/2403.16689](https://arxiv.org/abs/2403.16689)

    Synapse是一种神经符号化方法，旨在从有限演示中高效学习偏好概念，通过将偏好表示为神经符号程序并利用视觉解析、大型语言模型和程序合成相结合的方式来学习个人偏好。

    

    本文解决了偏好学习问题，旨在从视觉输入中学习用户特定偏好（例如，“好停车位”，“方便的下车位置”）。尽管与学习事实概念（例如，“红色立方体”）相似，但偏好学习是一个基本更加困难的问题，因为它涉及主观性质和个人特定训练数据的缺乏。我们使用一种名为Synapse的新框架来解决这个问题，这是一种神经符号化方法，旨在有效地从有限演示中学习偏好概念。Synapse将偏好表示为在图像上运作的领域特定语言（DSL）中的神经符号程序，并利用视觉解析、大型语言模型和程序合成的新组合来学习代表个人偏好的程序。我们通过广泛的实验评估了Synapse，包括一个关注与移动相关的用户案例研究。

    arXiv:2403.16689v1 Announce Type: cross  Abstract: This paper addresses the problem of preference learning, which aims to learn user-specific preferences (e.g., "good parking spot", "convenient drop-off location") from visual input. Despite its similarity to learning factual concepts (e.g., "red cube"), preference learning is a fundamentally harder problem due to its subjective nature and the paucity of person-specific training data. We address this problem using a new framework called Synapse, which is a neuro-symbolic approach designed to efficiently learn preferential concepts from limited demonstrations. Synapse represents preferences as neuro-symbolic programs in a domain-specific language (DSL) that operates over images, and leverages a novel combination of visual parsing, large language models, and program synthesis to learn programs representing individual preferences. We evaluate Synapse through extensive experimentation including a user case study focusing on mobility-related
    
[^2]: 自动驾驶中可解释人工智能的安全影响

    Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving

    [https://arxiv.org/abs/2403.12176](https://arxiv.org/abs/2403.12176)

    自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。

    

    末端到末端学习管道正在逐渐改变高度自主车辆的持续发展，这主要归功于深度学习的进步、大规模训练数据集的可用性以及综合传感器设备的改进。然而，当代学习方法在实时决策中缺乏可解释性，妨碍了用户的信任，并减弱了这类车辆的广泛部署和商业化。此外，当这些汽车参与或导致交通事故时，问题会变得更加严重。这种缺点从社会和法律的角度引起了严重的安全担忧。因此，在末端到末端自动驾驶中解释性是促进车辆自动化安全的关键。然而，当今最先进技术中研究人员通常将自动驾驶的安全性和可解释性方面分开研究。在本文中，我们旨在弥合这一差距

    arXiv:2403.12176v1 Announce Type: cross  Abstract: The end-to-end learning pipeline is gradually creating a paradigm shift in the ongoing development of highly autonomous vehicles, largely due to advances in deep learning, the availability of large-scale training datasets, and improvements in integrated sensor devices. However, a lack of interpretability in real-time decisions with contemporary learning methods impedes user trust and attenuates the widespread deployment and commercialization of such vehicles. Moreover, the issue is exacerbated when these cars are involved in or cause traffic accidents. Such drawback raises serious safety concerns from societal and legal perspectives. Consequently, explainability in end-to-end autonomous driving is essential to enable the safety of vehicular automation. However, the safety and explainability aspects of autonomous driving have generally been investigated disjointly by researchers in today's state of the art. In this paper, we aim to brid
    
[^3]: 注意错误！检测和定位视觉与语言导航中的指令错误

    Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation

    [https://arxiv.org/abs/2403.10700](https://arxiv.org/abs/2403.10700)

    提出了一个新的基准数据集，首次引入了各种类型的指令错误，考虑到潜在的人类原因，以评估连续环境中 VLN 系统的健壮性

    

    Vision-and-Language Navigation in Continuous Environments (VLN-CE) 是一项直观且具有挑战性的体验智能任务。代理人被要求通过执行一系列低级动作、遵循一系列自然语言指令来导航到目标目标。所有文献中的 VLN-CE 方法都假设语言指令是准确的。然而，在实践中，人类给出的指令可能由于不准确的记忆或混淆而包含空间环境描述中的错误。当前 VLN-CE 基准没有解决这种情况，使得 VLN-CE 中的最新方法在面对来自人类用户的错误指令时变得脆弱。我们首次提出了一个引入各种类型指令错误考虑潜在人类原因的新型基准数据集。该基准数据集为连续环境中的 VLN 系统的健壮性提供了宝贵的见解。我们观察到 noticeable...

    arXiv:2403.10700v1 Announce Type: cross  Abstract: Vision-and-Language Navigation in Continuous Environments (VLN-CE) is one of the most intuitive yet challenging embodied AI tasks. Agents are tasked to navigate towards a target goal by executing a set of low-level actions, following a series of natural language instructions. All VLN-CE methods in the literature assume that language instructions are exact. However, in practice, instructions given by humans can contain errors when describing a spatial environment due to inaccurate memory or confusion. Current VLN-CE benchmarks do not address this scenario, making the state-of-the-art methods in VLN-CE fragile in the presence of erroneous instructions from human users. For the first time, we propose a novel benchmark dataset that introduces various types of instruction errors considering potential human causes. This benchmark provides valuable insight into the robustness of VLN systems in continuous environments. We observe a noticeable 
    

