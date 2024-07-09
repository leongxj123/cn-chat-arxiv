# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Explore until Confident: Efficient Exploration for Embodied Question Answering](https://arxiv.org/abs/2403.15941) | 通过利用大型视觉-语言模型的语义推理能力，结合深度信息和视觉提示，提出了一种方法来解决具身问答中的有效探索和回答问题的挑战 |
| [^2] | [Blending Data-Driven Priors in Dynamic Games](https://arxiv.org/abs/2402.14174) | 探索一种在动态游戏中将数据驱动参考政策与基于优化博弈政策相融合的方法，提出了一种非合作动态博弈KLGame，其中包含了针对每个决策者的可调参数。 |
| [^3] | [Verifiably Following Complex Robot Instructions with Foundation Models](https://arxiv.org/abs/2402.11498) | 提出了一种名为语言指令地面化运动规划（LIMP）系统，利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，包括开放词汇参照和复杂的时空约束。 |
| [^4] | [Conditional Neural Expert Processes for Learning from Demonstration](https://arxiv.org/abs/2402.08424) | 条件神经专家过程（CNEP）是一种学习从演示中获取技能的新框架，通过将不同模式的演示分配给不同的专家网络，并利用潜在空间中的信息将专家与编码表示匹配，解决了相同技能演示的变化和多种方式获取的挑战。 |
| [^5] | [Lifelike Agility and Play on Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models.](http://arxiv.org/abs/2308.15143) | 该论文提出了一种使用生成模型和强化学习的框架，使四足机器人能够在复杂环境中像真实动物一样具有灵活性和策略。通过预训练生成模型，保留了动物行为的知识，并通过学习适应环境，克服挑战性的障碍。 |
| [^6] | [Neuro-Inspired Efficient Map Building via Fragmentation and Recall.](http://arxiv.org/abs/2307.05793) | 本文提出了一种神经启发的地图构建方法，通过分割和回溯来解决大型环境下的探索问题，并基于意外性的空间聚类设置探索子目标。 |

# 详细

[^1]: 探索直到自信: 面向具身问答的高效探索

    Explore until Confident: Efficient Exploration for Embodied Question Answering

    [https://arxiv.org/abs/2403.15941](https://arxiv.org/abs/2403.15941)

    通过利用大型视觉-语言模型的语义推理能力，结合深度信息和视觉提示，提出了一种方法来解决具身问答中的有效探索和回答问题的挑战

    

    我们考虑了具身问答（EQA）的问题，这指的是在需要主动探索环境以收集信息直到对问题的答案有自信的具身代理，例如机器人。在这项工作中，我们利用大规模视觉-语言模型（VLMs）的强大语义推理能力来高效探索和回答这些问题。然而，在EQA中使用VLMs时存在两个主要挑战：它们没有内部记忆将场景映射以便规划如何随时间探索，并且它们的置信度可能被错误校准并可能导致机器人过早停止探索或过度探索。我们提出了一种方法，首先基于深度信息和通过视觉提示VLM来构建场景的语义地图-利用其对场景相关区域的广泛知识来进行探索。接下来，我们使用符合预测来校准VLM的置信度。

    arXiv:2403.15941v1 Announce Type: cross  Abstract: We consider the problem of Embodied Question Answering (EQA), which refers to settings where an embodied agent such as a robot needs to actively explore an environment to gather information until it is confident about the answer to a question. In this work, we leverage the strong semantic reasoning capabilities of large vision-language models (VLMs) to efficiently explore and answer such questions. However, there are two main challenges when using VLMs in EQA: they do not have an internal memory for mapping the scene to be able to plan how to explore over time, and their confidence can be miscalibrated and can cause the robot to prematurely stop exploration or over-explore. We propose a method that first builds a semantic map of the scene based on depth information and via visual prompting of a VLM - leveraging its vast knowledge of relevant regions of the scene for exploration. Next, we use conformal prediction to calibrate the VLM's 
    
[^2]: 在动态游戏中融合数据驱动的先验知识

    Blending Data-Driven Priors in Dynamic Games

    [https://arxiv.org/abs/2402.14174](https://arxiv.org/abs/2402.14174)

    探索一种在动态游戏中将数据驱动参考政策与基于优化博弈政策相融合的方法，提出了一种非合作动态博弈KLGame，其中包含了针对每个决策者的可调参数。

    

    随着智能机器人如自动驾驶车辆在人群中的部署越来越多，这些系统应该在安全的、与人互动意识相关的运动规划中利用基于模型的博弈论规划器与数据驱动政策的程度仍然是一个悬而未决的问题。本文探讨了一种融合数据驱动参考政策和基于优化的博弈论政策的原则性方法。我们制定了KLGame，这是一种带有Kullback-Leibler（KL）正则化的非合作动态博弈，针对一个一般的、随机的，可能是多模式的参考政策。

    arXiv:2402.14174v1 Announce Type: cross  Abstract: As intelligent robots like autonomous vehicles become increasingly deployed in the presence of people, the extent to which these systems should leverage model-based game-theoretic planners versus data-driven policies for safe, interaction-aware motion planning remains an open question. Existing dynamic game formulations assume all agents are task-driven and behave optimally. However, in reality, humans tend to deviate from the decisions prescribed by these models, and their behavior is better approximated under a noisy-rational paradigm. In this work, we investigate a principled methodology to blend a data-driven reference policy with an optimization-based game-theoretic policy. We formulate KLGame, a type of non-cooperative dynamic game with Kullback-Leibler (KL) regularization with respect to a general, stochastic, and possibly multi-modal reference policy. Our method incorporates, for each decision maker, a tunable parameter that pe
    
[^3]: 使用基础模型可验证地遵循复杂机器人指令

    Verifiably Following Complex Robot Instructions with Foundation Models

    [https://arxiv.org/abs/2402.11498](https://arxiv.org/abs/2402.11498)

    提出了一种名为语言指令地面化运动规划（LIMP）系统，利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，包括开放词汇参照和复杂的时空约束。

    

    让机器人能够遵循复杂的自然语言指令是一个重要但具有挑战性的问题。人们希望在指导机器人时能够灵活表达约束，指向任意地标并验证行为。相反，机器人必须将人类指令消除歧义，将指令参照物联系到真实世界中。我们提出了一种名为语言指令地面化运动规划（LIMP）的系统，该系统利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，涵盖了开放词汇参照和复杂的时空约束。与先前在机器人任务执行中使用基础模型的方法相比，LIMP构建了一个可解释的指令表示，揭示了机器人与指导者预期动机的一致性，并实现了机器人行为的综合。

    arXiv:2402.11498v1 Announce Type: cross  Abstract: Enabling robots to follow complex natural language instructions is an important yet challenging problem. People want to flexibly express constraints, refer to arbitrary landmarks and verify behavior when instructing robots. Conversely, robots must disambiguate human instructions into specifications and ground instruction referents in the real world. We propose Language Instruction grounding for Motion Planning (LIMP), a system that leverages foundation models and temporal logics to generate instruction-conditioned semantic maps that enable robots to verifiably follow expressive and long-horizon instructions with open vocabulary referents and complex spatiotemporal constraints. In contrast to prior methods for using foundation models in robot task execution, LIMP constructs an explainable instruction representation that reveals the robot's alignment with an instructor's intended motives and affords the synthesis of robot behaviors that 
    
[^4]: 条件神经专家过程用于从演示中学习

    Conditional Neural Expert Processes for Learning from Demonstration

    [https://arxiv.org/abs/2402.08424](https://arxiv.org/abs/2402.08424)

    条件神经专家过程（CNEP）是一种学习从演示中获取技能的新框架，通过将不同模式的演示分配给不同的专家网络，并利用潜在空间中的信息将专家与编码表示匹配，解决了相同技能演示的变化和多种方式获取的挑战。

    

    从演示中学习（LfD）是机器人学中广泛使用的一种技术，用于技能获取。然而，相同技能的演示可能存在显著的变化，或者学习系统可能同时尝试获取相同技能的不同方式，这使得将这些动作编码为运动原语变得具有挑战性。为了解决这些挑战，我们提出了一个LfD框架，即条件神经专家过程（CNEP），它学习将来自不同模式的演示分配给不同的专家网络，利用潜在空间中的内在信息将专家与编码表示匹配起来。CNEP不需要在哪种模式下轨迹属于的监督。在人工生成的数据集上进行的实验证明了CNEP的有效性。此外，我们将CNEP与另一个LfD框架——条件神经运动原语（CNMP）在一系列任务上的性能进行了比较，包括对真实机器人进行实验。

    Learning from Demonstration (LfD) is a widely used technique for skill acquisition in robotics. However, demonstrations of the same skill may exhibit significant variances, or learning systems may attempt to acquire different means of the same skill simultaneously, making it challenging to encode these motions into movement primitives. To address these challenges, we propose an LfD framework, namely the Conditional Neural Expert Processes (CNEP), that learns to assign demonstrations from different modes to distinct expert networks utilizing the inherent information within the latent space to match experts with the encoded representations. CNEP does not require supervision on which mode the trajectories belong to. Provided experiments on artificially generated datasets demonstrate the efficacy of CNEP. Furthermore, we compare the performance of CNEP with another LfD framework, namely Conditional Neural Movement Primitives (CNMP), on a range of tasks, including experiments on a real robo
    
[^5]: 使用强化学习和生成预训练模型在四足机器人上实现生动的灵活性和游戏性

    Lifelike Agility and Play on Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models. (arXiv:2308.15143v1 [cs.RO])

    [http://arxiv.org/abs/2308.15143](http://arxiv.org/abs/2308.15143)

    该论文提出了一种使用生成模型和强化学习的框架，使四足机器人能够在复杂环境中像真实动物一样具有灵活性和策略。通过预训练生成模型，保留了动物行为的知识，并通过学习适应环境，克服挑战性的障碍。

    

    总结动物和人类的知识启发了机器人创新。在这项工作中，我们提出了一种框架，使四足机器人能够在复杂环境中像真实动物一样拥有生动的灵活性和策略。受到在语言和图像理解方面表现出色的大型预训练模型的启发，我们引入了先进的深度生成模型的能力，以生成模拟真实动物动作的运动控制信号。与传统控制器和端到端强化学习方法只针对特定任务不同，我们提出在动物运动数据集上预训练生成模型，以保留有表达力的动物行为知识。预训练模型拥有足够的原始级知识，但与环境无关。然后，在学习的后续阶段，通过穿越一些以前的方法很少考虑的具有挑战性的障碍，如穿过狭窄的空间等，使其适应环境。

    Summarizing knowledge from animals and human beings inspires robotic innovations. In this work, we propose a framework for driving legged robots act like real animals with lifelike agility and strategy in complex environments. Inspired by large pre-trained models witnessed with impressive performance in language and image understanding, we introduce the power of advanced deep generative models to produce motor control signals stimulating legged robots to act like real animals. Unlike conventional controllers and end-to-end RL methods that are task-specific, we propose to pre-train generative models over animal motion datasets to preserve expressive knowledge of animal behavior. The pre-trained model holds sufficient primitive-level knowledge yet is environment-agnostic. It is then reused for a successive stage of learning to align with the environments by traversing a number of challenging obstacles that are rarely considered in previous approaches, including creeping through narrow sp
    
[^6]: 神经启发的高效地图构建通过分割和回溯

    Neuro-Inspired Efficient Map Building via Fragmentation and Recall. (arXiv:2307.05793v1 [cs.AI])

    [http://arxiv.org/abs/2307.05793](http://arxiv.org/abs/2307.05793)

    本文提出了一种神经启发的地图构建方法，通过分割和回溯来解决大型环境下的探索问题，并基于意外性的空间聚类设置探索子目标。

    

    动物和机器人通过构建和完善空间地图来导航环境。这些地图使得包括回家、规划、搜索和觅食在内的功能成为可能。在大型环境中，探索空间是一个难题：代理可能会陷入局部区域。在这里，我们从神经科学中汲取经验，提出并应用了分割和回溯（FarMap）的概念。代理通过基于意外性的空间聚类来解决地图构建问题，同时将其用于设置空间探索的子目标。代理构建和使用本地地图来预测他们的观测结果；高意外性会导致“分割事件”，从而截断本地地图。在这些事件中，最近的本地地图被放入长期记忆（LTM）中，并初始化另一个本地地图。如果断裂点的观察结果与存储的某个本地地图的观察结果相匹配，那么该地图就会被回溯（并重用）自LTM。分割点诱导.

    Animals and robots navigate through environments by building and refining maps of the space. These maps enable functions including navigating back to home, planning, search, and foraging. In large environments, exploration of the space is a hard problem: agents can become stuck in local regions. Here, we use insights from neuroscience to propose and apply the concept of Fragmentation-and-Recall (FarMap), with agents solving the mapping problem by building local maps via a surprisal-based clustering of space, which they use to set subgoals for spatial exploration. Agents build and use a local map to predict their observations; high surprisal leads to a ``fragmentation event'' that truncates the local map. At these events, the recent local map is placed into long-term memory (LTM), and a different local map is initialized. If observations at a fracture point match observations in one of the stored local maps, that map is recalled (and thus reused) from LTM. The fragmentation points induc
    

