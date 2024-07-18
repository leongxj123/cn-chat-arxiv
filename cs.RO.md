# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following](https://arxiv.org/abs/2402.06559) | 本文提出了一种Diffusion-ES方法，它结合了无梯度优化和轨迹去噪技术，用于优化黑盒非可微目标。该方法通过从扩散模型中采样轨迹，并使用黑盒奖励函数对其进行评分，实现了更高的多样性和可解释性。 |
| [^2] | [NOD-TAMP: Multi-Step Manipulation Planning with Neural Object Descriptors.](http://arxiv.org/abs/2311.01530) | NOD-TAMP是一个基于TAMP的框架，利用神经物体描述符来解决复杂操纵任务中的泛化问题，通过从少量人类演示中提取轨迹并进行调整，有效解决了长时程任务的挑战，并在模拟环境中优于现有方法。 |
| [^3] | [Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review.](http://arxiv.org/abs/2308.05731) | 这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。 |

# 详细

[^1]: Diffusion-ES:基于扩散的零梯度规划用于自动驾驶和零阶指令跟随

    Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following

    [https://arxiv.org/abs/2402.06559](https://arxiv.org/abs/2402.06559)

    本文提出了一种Diffusion-ES方法，它结合了无梯度优化和轨迹去噪技术，用于优化黑盒非可微目标。该方法通过从扩散模型中采样轨迹，并使用黑盒奖励函数对其进行评分，实现了更高的多样性和可解释性。

    

    扩散模型在决策和控制中对复杂和多模态轨迹分布建模有很强优势。最近提出了奖励梯度引导去噪方法，用于产生在扩散模型所捕获的数据分布下，同时最大化可微分奖励函数和似然性的轨迹。奖励梯度引导去噪需要一个适合于清洁和噪声样本的可微分奖励函数，从而限制了其作为一种通用轨迹优化器的适用性。在本文中，我们提出了DiffusionES，一种将无梯度优化和轨迹去噪相结合的方法，用于在数据流形中优化黑盒非可微目标。Diffusion-ES从扩散模型中采样轨迹，并使用黑盒奖励函数对其进行评分。它通过截断扩散过程对得分高的轨迹进行变异，该过程应用少量的噪声和去噪步骤，从而实现了更高的多样性和更好的可解释性。

    Diffusion models excel at modeling complex and multimodal trajectory distributions for decision-making and control. Reward-gradient guided denoising has been recently proposed to generate trajectories that maximize both a differentiable reward function and the likelihood under the data distribution captured by a diffusion model. Reward-gradient guided denoising requires a differentiable reward function fitted to both clean and noised samples, limiting its applicability as a general trajectory optimizer. In this paper, we propose DiffusionES, a method that combines gradient-free optimization with trajectory denoising to optimize black-box non-differentiable objectives while staying in the data manifold. Diffusion-ES samples trajectories during evolutionary search from a diffusion model and scores them using a black-box reward function. It mutates high-scoring trajectories using a truncated diffusion process that applies a small number of noising and denoising steps, allowing for much mo
    
[^2]: NOD-TAMP:多步骤操纵规划中的神经物体描述符

    NOD-TAMP: Multi-Step Manipulation Planning with Neural Object Descriptors. (arXiv:2311.01530v1 [cs.RO])

    [http://arxiv.org/abs/2311.01530](http://arxiv.org/abs/2311.01530)

    NOD-TAMP是一个基于TAMP的框架，利用神经物体描述符来解决复杂操纵任务中的泛化问题，通过从少量人类演示中提取轨迹并进行调整，有效解决了长时程任务的挑战，并在模拟环境中优于现有方法。

    

    在家居和工厂环境中开发复杂操纵任务的智能机器人仍然具有挑战性，因为长时程任务、接触丰富的操纵以及需要在各种物体形状和场景布局之间进行泛化。虽然任务和运动规划（TAMP）提供了一个有希望的解决方案，但是它的假设，如动力学模型，限制了它在新颖背景中的适应性。神经物体描述符（NODs）在物体和场景泛化方面显示出了潜力，但在处理更广泛任务方面存在局限性。我们提出的基于TAMP的框架NOD-TAMP从少数人类演示中提取短的操纵轨迹，使用NOD特征来调整这些轨迹，并组合它们来解决广泛的长时程任务。在模拟环境中验证后，NOD-TAMP有效应对各种挑战，优于现有方法，建立了一个强有力的操纵规划框架。

    Developing intelligent robots for complex manipulation tasks in household and factory settings remains challenging due to long-horizon tasks, contact-rich manipulation, and the need to generalize across a wide variety of object shapes and scene layouts. While Task and Motion Planning (TAMP) offers a promising solution, its assumptions such as kinodynamic models limit applicability in novel contexts. Neural object descriptors (NODs) have shown promise in object and scene generalization but face limitations in addressing broader tasks. Our proposed TAMP-based framework, NOD-TAMP, extracts short manipulation trajectories from a handful of human demonstrations, adapts these trajectories using NOD features, and composes them to solve broad long-horizon tasks. Validated in a simulation environment, NOD-TAMP effectively tackles varied challenges and outperforms existing methods, establishing a cohesive framework for manipulation planning. For videos and other supplemental material, see the pr
    
[^3]: 重新思考基于深度学习的自动驾驶系统中的预测和规划的整合：一项综述

    Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review. (arXiv:2308.05731v1 [cs.RO])

    [http://arxiv.org/abs/2308.05731](http://arxiv.org/abs/2308.05731)

    这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。

    

    自动驾驶有可能彻底改变个人、公共和货运交通的方式。除了感知环境的巨大挑战外，即准确地使用可用的传感器数据感知环境，自动驾驶还包括规划一个安全、舒适和高效的运动轨迹。为了促进安全和进步，许多工作依赖于模块化的交通未来运动的预测。模块化的自动驾驶系统通常将预测和规划作为顺序的独立任务处理。虽然这考虑了周围交通对自车的影响，但它未能预测交通参与者对自车行为的反应。最近的研究表明，将预测和规划整合为相互依赖的联合步骤是实现安全、高效和舒适驾驶所必需的。虽然有各种模型实现了这种集成系统，但对不同原理的全面概述和理论理解仍然缺乏。

    Automated driving has the potential to revolutionize personal, public, and freight mobility. Besides the enormous challenge of perception, i.e. accurately perceiving the environment using available sensor data, automated driving comprises planning a safe, comfortable, and efficient motion trajectory. To promote safety and progress, many works rely on modules that predict the future motion of surrounding traffic. Modular automated driving systems commonly handle prediction and planning as sequential separate tasks. While this accounts for the influence of surrounding traffic on the ego-vehicle, it fails to anticipate the reactions of traffic participants to the ego-vehicle's behavior. Recent works suggest that integrating prediction and planning in an interdependent joint step is necessary to achieve safe, efficient, and comfortable driving. While various models implement such integrated systems, a comprehensive overview and theoretical understanding of different principles are lacking.
    

