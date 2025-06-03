# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation](https://arxiv.org/abs/2402.17645) | SongComposer提出了一种用于歌曲生成的大型语言模型，采用符号化的歌曲表示，实现了LLM可以明确创作歌曲的能力。 |

# 详细

[^1]: SongComposer：一种用于歌曲生成的大型语言模型，用于歌词和旋律创作

    SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation

    [https://arxiv.org/abs/2402.17645](https://arxiv.org/abs/2402.17645)

    SongComposer提出了一种用于歌曲生成的大型语言模型，采用符号化的歌曲表示，实现了LLM可以明确创作歌曲的能力。

    

    我们提出了SongComposer，一个为歌曲创作而设计的创新型LLM。它能够理解和生成歌曲中的旋律和歌词，通过利用LLM的能力在符号化的歌曲表示中生成。现有的与音乐相关的LLM将音乐视为量化的音频信号，而这种隐式编码导致了编码效率低下和灵活性差。相比之下，我们采用了符号化的歌曲表示，这是人类为音乐设计的成熟和高效的方式，并使LLM能够像人类一样明确地创作歌曲。在实践中，我们设计了一种新颖的元组设计，用于格式化歌词和旋律中的三个音符属性（音高、持续时间和休止时间），从而保证LLM对音乐符号的正确理解，并实现歌词和旋律之间的精确对齐。为了向LLM灌输基本的音乐理解，我们精心收集了SongCompose-PT，一个大规模的歌曲预训练数据集，其中包括了歌词、旋律和成对的

    arXiv:2402.17645v1 Announce Type: cross  Abstract: We present SongComposer, an innovative LLM designed for song composition. It could understand and generate melodies and lyrics in symbolic song representations, by leveraging the capability of LLM. Existing music-related LLM treated the music as quantized audio signals, while such implicit encoding leads to inefficient encoding and poor flexibility. In contrast, we resort to symbolic song representation, the mature and efficient way humans designed for music, and enable LLM to explicitly compose songs like humans. In practice, we design a novel tuple design to format lyric and three note attributes (pitch, duration, and rest duration) in the melody, which guarantees the correct LLM understanding of musical symbols and realizes precise alignment between lyrics and melody. To impart basic music understanding to LLM, we carefully collected SongCompose-PT, a large-scale song pretraining dataset that includes lyrics, melodies, and paired ly
    

