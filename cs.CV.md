# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synapse: Learning Preferential Concepts from Visual Demonstrations](https://arxiv.org/abs/2403.16689) | Synapse是一种神经符号化方法，旨在从有限演示中高效学习偏好概念，通过将偏好表示为神经符号程序并利用视觉解析、大型语言模型和程序合成相结合的方式来学习个人偏好。 |
| [^2] | [Expressive Text-to-Image Generation with Rich Text.](http://arxiv.org/abs/2304.06720) | 本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。 |

# 详细

[^1]: Synapse: 从视觉演示中学习优先概念

    Synapse: Learning Preferential Concepts from Visual Demonstrations

    [https://arxiv.org/abs/2403.16689](https://arxiv.org/abs/2403.16689)

    Synapse是一种神经符号化方法，旨在从有限演示中高效学习偏好概念，通过将偏好表示为神经符号程序并利用视觉解析、大型语言模型和程序合成相结合的方式来学习个人偏好。

    

    本文解决了偏好学习问题，旨在从视觉输入中学习用户特定偏好（例如，“好停车位”，“方便的下车位置”）。尽管与学习事实概念（例如，“红色立方体”）相似，但偏好学习是一个基本更加困难的问题，因为它涉及主观性质和个人特定训练数据的缺乏。我们使用一种名为Synapse的新框架来解决这个问题，这是一种神经符号化方法，旨在有效地从有限演示中学习偏好概念。Synapse将偏好表示为在图像上运作的领域特定语言（DSL）中的神经符号程序，并利用视觉解析、大型语言模型和程序合成的新组合来学习代表个人偏好的程序。我们通过广泛的实验评估了Synapse，包括一个关注与移动相关的用户案例研究。

    arXiv:2403.16689v1 Announce Type: cross  Abstract: This paper addresses the problem of preference learning, which aims to learn user-specific preferences (e.g., "good parking spot", "convenient drop-off location") from visual input. Despite its similarity to learning factual concepts (e.g., "red cube"), preference learning is a fundamentally harder problem due to its subjective nature and the paucity of person-specific training data. We address this problem using a new framework called Synapse, which is a neuro-symbolic approach designed to efficiently learn preferential concepts from limited demonstrations. Synapse represents preferences as neuro-symbolic programs in a domain-specific language (DSL) that operates over images, and leverages a novel combination of visual parsing, large language models, and program synthesis to learn programs representing individual preferences. We evaluate Synapse through extensive experimentation including a user case study focusing on mobility-related
    
[^2]: 富文本生成表达性文本图像

    Expressive Text-to-Image Generation with Rich Text. (arXiv:2304.06720v1 [cs.CV])

    [http://arxiv.org/abs/2304.06720](http://arxiv.org/abs/2304.06720)

    本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。

    

    纯文本已经成为文字到图像合成的流行界面。但是，它的定制选项有限，阻碍了用户精确描述所需的输出。为了解决这些挑战，我们提出使用支持字体样式、大小、颜色和脚注等格式的富文本编辑器。我们从富文本中提取每个字的属性，以启用局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成。我们通过基于区域的扩散过程实现了这些功能。我们的实验表明，我们的方法可以比现有的文本到图像方法更好地生成高质量和多样化的图像。

    Plain text has become a prevalent interface for text-to-image synthesis. However, its limited customization options hinder users from accurately describing desired outputs. For example, plain text makes it hard to specify continuous quantities, such as the precise RGB color value or importance of each word. Furthermore, creating detailed text prompts for complex scenes is tedious for humans to write and challenging for text encoders to interpret. To address these challenges, we propose using a rich-text editor supporting formats such as font style, size, color, and footnote. We extract each word's attributes from rich text to enable local style control, explicit token reweighting, precise color rendering, and detailed region synthesis. We achieve these capabilities through a region-based diffusion process. We first obtain each word's region based on cross-attention maps of a vanilla diffusion process using plain text. For each region, we enforce its text attributes by creating region-s
    

