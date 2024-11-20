# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AccoMontage-3: Full-Band Accompaniment Arrangement via Sequential Style Transfer and Multi-Track Function Prior.](http://arxiv.org/abs/2310.16334) | AccoMontage-3是一种通过顺序风格转换和多轨道功能先验实现全音乐伴奏编排的系统，可以根据主旋律与和弦的输入生成多音轨的伴奏。 |

# 详细

[^1]: AccoMontage-3: 通过顺序风格转换和多轨道功能先验实现全音乐伴奏编排

    AccoMontage-3: Full-Band Accompaniment Arrangement via Sequential Style Transfer and Multi-Track Function Prior. (arXiv:2310.16334v1 [cs.SD])

    [http://arxiv.org/abs/2310.16334](http://arxiv.org/abs/2310.16334)

    AccoMontage-3是一种通过顺序风格转换和多轨道功能先验实现全音乐伴奏编排的系统，可以根据主旋律与和弦的输入生成多音轨的伴奏。

    

    我们提出了AccoMontage-3，这是一个符号音乐自动化系统，可以根据主旋律与和弦的输入（即引导乐谱），生成多音轨的全音乐伴奏。该系统包含三个模块化组件，每个组件模拟全音乐作曲的重要方面。第一个组件是钢琴编曲师，通过将纹理风格转换为和弦，使用潜在的和弦-纹理分离和启发式纹理供应者检索，生成钢琴伴奏。第二个组件根据个别音轨功能编码的管弦乐风格，将钢琴伴奏乐谱编排成全音乐伴奏。将前两个组件连接起来的第三个组件是一个先验模型，用于描述整首音乐作品上的编曲风格的全局结构。整个系统以端到端的方式自我监督地学习生成全音乐伴奏，将风格转换应用于两个层面的多声部协调。

    We propose AccoMontage-3, a symbolic music automation system capable of generating multi-track, full-band accompaniment based on the input of a lead melody with chords (i.e., a lead sheet). The system contains three modular components, each modelling a vital aspect of full-band composition. The first component is a piano arranger that generates piano accompaniment for the lead sheet by transferring texture styles to the chords using latent chord-texture disentanglement and heuristic retrieval of texture donors. The second component orchestrates the piano accompaniment score into full-band arrangement according to the orchestration style encoded by individual track functions. The third component, which connects the previous two, is a prior model characterizing the global structure of orchestration style over the whole piece of music. From end to end, the system learns to generate full-band accompaniment in a self-supervised fashion, applying style transfer at two levels of polyphonic co
    

