# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis.](http://arxiv.org/abs/2312.10741) | StyleSinger是针对领域外演唱声音合成的风格转移模型，通过残差风格适配器（RSA）捕捉多样的风格特征实现高质量的合成演唱声音。 |

# 详细

[^1]: StyleSinger: 针对领域外演唱声音合成的风格转移

    StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis. (arXiv:2312.10741v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2312.10741](http://arxiv.org/abs/2312.10741)

    StyleSinger是针对领域外演唱声音合成的风格转移模型，通过残差风格适配器（RSA）捕捉多样的风格特征实现高质量的合成演唱声音。

    

    针对领域外演唱声音合成（SVS）的风格转移专注于生成高质量的演唱声音，该声音具有从参考演唱声音样本中衍生的未见风格（如音色、情感、发音和发音技巧）。然而，模拟演唱声音风格的精细差异是一项艰巨的任务，因为演唱声音具有非常高的表现力。此外，现有的SVS方法在领域外场景中合成的演唱声音质量下降，因为它们基于训练阶段可辨别出目标声音属性的假设。为了克服这些挑战，我们提出了StyleSinger，这是第一个用于领域外参考演唱声音样本的零样式转移的演唱声音合成模型。StyleSinger采用了两种关键方法以提高效果：1）残差风格适配器（RSA），它使用残差量化模块来捕捉多样的风格特征。

    Style transfer for out-of-domain (OOD) singing voice synthesis (SVS) focuses on generating high-quality singing voices with unseen styles (such as timbre, emotion, pronunciation, and articulation skills) derived from reference singing voice samples. However, the endeavor to model the intricate nuances of singing voice styles is an arduous task, as singing voices possess a remarkable degree of expressiveness. Moreover, existing SVS methods encounter a decline in the quality of synthesized singing voices in OOD scenarios, as they rest upon the assumption that the target vocal attributes are discernible during the training phase. To overcome these challenges, we propose StyleSinger, the first singing voice synthesis model for zero-shot style transfer of out-of-domain reference singing voice samples. StyleSinger incorporates two critical approaches for enhanced effectiveness: 1) the Residual Style Adaptor (RSA) which employs a residual quantization module to capture diverse style character
    

