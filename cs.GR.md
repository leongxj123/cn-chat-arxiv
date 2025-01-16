# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Expressive Text-to-Image Generation with Rich Text.](http://arxiv.org/abs/2304.06720) | 本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。 |

# 详细

[^1]: 富文本生成表达性文本图像

    Expressive Text-to-Image Generation with Rich Text. (arXiv:2304.06720v1 [cs.CV])

    [http://arxiv.org/abs/2304.06720](http://arxiv.org/abs/2304.06720)

    本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。

    

    纯文本已经成为文字到图像合成的流行界面。但是，它的定制选项有限，阻碍了用户精确描述所需的输出。为了解决这些挑战，我们提出使用支持字体样式、大小、颜色和脚注等格式的富文本编辑器。我们从富文本中提取每个字的属性，以启用局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成。我们通过基于区域的扩散过程实现了这些功能。我们的实验表明，我们的方法可以比现有的文本到图像方法更好地生成高质量和多样化的图像。

    Plain text has become a prevalent interface for text-to-image synthesis. However, its limited customization options hinder users from accurately describing desired outputs. For example, plain text makes it hard to specify continuous quantities, such as the precise RGB color value or importance of each word. Furthermore, creating detailed text prompts for complex scenes is tedious for humans to write and challenging for text encoders to interpret. To address these challenges, we propose using a rich-text editor supporting formats such as font style, size, color, and footnote. We extract each word's attributes from rich text to enable local style control, explicit token reweighting, precise color rendering, and detailed region synthesis. We achieve these capabilities through a region-based diffusion process. We first obtain each word's region based on cross-attention maps of a vanilla diffusion process using plain text. For each region, we enforce its text attributes by creating region-s
    

