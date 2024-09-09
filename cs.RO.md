# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CURE: Simulation-Augmented Auto-Tuning in Robotics](https://arxiv.org/abs/2402.05399) | 本论文提出了一种模拟辅助的自动调节技术，用于解决机器人系统中的高度可配置参数的优化问题。该技术通过解决软硬件之间配置选项的交互问题，实现了在不同环境和机器人平台之间的性能迁移。 |

# 详细

[^1]: CURE: 机器人领域的模拟辅助自动调节技术

    CURE: Simulation-Augmented Auto-Tuning in Robotics

    [https://arxiv.org/abs/2402.05399](https://arxiv.org/abs/2402.05399)

    本论文提出了一种模拟辅助的自动调节技术，用于解决机器人系统中的高度可配置参数的优化问题。该技术通过解决软硬件之间配置选项的交互问题，实现了在不同环境和机器人平台之间的性能迁移。

    

    机器人系统通常由多个子系统组成，例如定位和导航，每个子系统又包含许多可配置的组件（例如选择不同的规划算法）。一旦选择了某个算法，就需要设置相关的配置选项以达到适当的值。系统堆栈中的配置选项会产生复杂的交互关系。在高度可配置的机器人中找到最佳配置来实现期望的性能是一个重大挑战，因为软件和硬件之间的配置选项交互导致了庞大且复杂的配置空间。性能迁移在不同的环境和机器人平台之间也是一个难题。数据高效优化算法（例如贝叶斯优化）已越来越多地用于自动化调整网络物理系统中的可配置参数。然而，这样的优化算法在机器人领域应用仍有局限性。

    Robotic systems are typically composed of various subsystems, such as localization and navigation, each encompassing numerous configurable components (e.g., selecting different planning algorithms). Once an algorithm has been selected for a component, its associated configuration options must be set to the appropriate values. Configuration options across the system stack interact non-trivially. Finding optimal configurations for highly configurable robots to achieve desired performance poses a significant challenge due to the interactions between configuration options across software and hardware that result in an exponentially large and complex configuration space. These challenges are further compounded by the need for transferability between different environments and robotic platforms. Data efficient optimization algorithms (e.g., Bayesian optimization) have been increasingly employed to automate the tuning of configurable parameters in cyber-physical systems. However, such optimiz
    

