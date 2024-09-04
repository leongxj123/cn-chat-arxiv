# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching.](http://arxiv.org/abs/2309.05027) | VoiceFlow使用矫正流匹配算法实现了高效文本转语音，并在合成质量上优于传统的扩散模型。 |

# 详细

[^1]: VoiceFlow: 使用矫正流匹配的高效文本转语音

    VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching. (arXiv:2309.05027v1 [eess.AS])

    [http://arxiv.org/abs/2309.05027](http://arxiv.org/abs/2309.05027)

    VoiceFlow使用矫正流匹配算法实现了高效文本转语音，并在合成质量上优于传统的扩散模型。

    

    尽管扩散模型在文本转语音中因其强大的生成能力而成为一种流行选择，但从扩散模型中进行采样的内在复杂性损害了其效率。相反，我们提出了VoiceFlow，一种利用矫正流匹配算法来实现高合成质量的声学模型，只需有限次采样步骤即可实现。VoiceFlow将生成mel-spectrograms的过程转化为一个普通微分方程，在文本输入的条件下进行求解，并估计出其向量场。然后，矫正流技术有效地使其采样轨迹直线化，实现高效合成。在单个和多个说话者语料库上进行的主观和客观评估显示，VoiceFlow相对于扩散模型具有更优异的合成质量。消融研究进一步验证了VoiceFlow中矫正流技术的有效性。

    Although diffusion models in text-to-speech have become a popular choice due to their strong generative ability, the intrinsic complexity of sampling from diffusion models harms their efficiency. Alternatively, we propose VoiceFlow, an acoustic model that utilizes a rectified flow matching algorithm to achieve high synthesis quality with a limited number of sampling steps. VoiceFlow formulates the process of generating mel-spectrograms into an ordinary differential equation conditional on text inputs, whose vector field is then estimated. The rectified flow technique then effectively straightens its sampling trajectory for efficient synthesis. Subjective and objective evaluations on both single and multi-speaker corpora showed the superior synthesis quality of VoiceFlow compared to the diffusion counterpart. Ablation studies further verified the validity of the rectified flow technique in VoiceFlow.
    

