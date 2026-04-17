"""
demo_data.py — Generate synthetic research paper PDFs for demo purposes.

Run:
    python demo_data.py

Creates sample_docs/attention_is_all_you_need_mock.pdf and others.
Uses only stdlib + reportlab (optional). Falls back to plain text .txt if reportlab missing.
"""

import os
from pathlib import Path

SAMPLE_PAPERS = [
    {
        "filename": "attention_transformer_2017.txt",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "type": "AI / ML",
        "content": """
Attention Is All You Need
Vaswani et al., 2017
URL: https://arxiv.org/abs/1706.03762

Abstract:
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks
that include an encoder and a decoder. The best performing models also connect the encoder and decoder
through an attention mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

Introduction:
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular,
have been firmly established as state of the art approaches in sequence modeling and transduction problems
such as language modeling and machine translation. Numerous efforts have since continued to push
the boundaries of recurrent language models and encoder-decoder architectures.

The Transformer:
The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise,
fully connected layers for both the encoder and decoder. The encoder maps an input sequence of symbol
representations to a sequence of continuous representations. Given z, the decoder then generates
an output sequence of symbols one element at a time.

Multi-Head Attention:
Instead of performing a single attention function with d_model-dimensional keys, values and queries,
we found it beneficial to linearly project the queries, keys and values h times with different,
learned linear projections to dk, dk and dv dimensions, respectively.

Results:
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best
previously reported models including ensembles by more than 2.0 BLEU, establishing a new state-of-the-art
BLEU score of 28.4. The big transformer model achieves 41.0 BLEU on the WMT 2014 English-to-French
translation task, outperforming all of the previously published single models, at less than 1/4 the
training cost of the previous state-of-the-art model.

Conclusion:
In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention. The Transformer can be trained significantly faster than architectures
based on recurrent or convolutional layers.
""",
    },
    {
        "filename": "bert_pretraining_2018.txt",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/abs/1810.04805",
        "type": "AI / ML",
        "content": """
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Devlin et al., 2018
URL: https://arxiv.org/abs/1810.04805

Abstract:
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder
Representations from Transformers. Unlike recent language representation models, BERT is designed to
pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left
and right context in all layers.

Pre-training BERT:
We pre-train BERT using two unsupervised tasks. Task 1 is Masked LM. A standard conditional language
model can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow
each word to indirectly see itself, and the model could trivially predict the target word in a
multi-layered context.

Task 2 - Next Sentence Prediction:
Many important downstream tasks such as Question Answering and Natural Language Inference are based on
understanding the relationship between two sentences, which is not directly captured by language modeling.
In order to train a model that understands sentence relationships, we pre-train for a binarized next
sentence prediction task.

Fine-tuning BERT:
Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to
model many downstream tasks by swapping out the appropriate inputs and outputs. For each task, we
simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters
end-to-end.

Results:
BERT obtains new state-of-the-art results on eleven natural language processing tasks, including
pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7%
(4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute
improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
""",
    },
    {
        "filename": "llm_security_survey_2024.txt",
        "title": "Security of Large Language Models: A Survey of Attacks and Defenses",
        "url": "https://arxiv.org/abs/2402.00888",
        "type": "Security",
        "content": """
Security of Large Language Models: A Survey of Attacks and Defenses
Smith et al., 2024
URL: https://arxiv.org/abs/2402.00888

Abstract:
Large language models (LLMs) have demonstrated remarkable capabilities across various applications,
but their widespread deployment raises significant security concerns. This survey provides a comprehensive
overview of security vulnerabilities in LLMs, including prompt injection attacks, jailbreaking techniques,
data poisoning, and model extraction attacks, along with corresponding defense mechanisms.

Prompt Injection Attacks:
Prompt injection is a class of attacks where adversarial instructions are embedded in user inputs
to manipulate LLM behavior. Direct prompt injection involves users crafting inputs that override
the system prompt, while indirect prompt injection occurs when malicious instructions are embedded
in external content that the LLM retrieves and processes.

Jailbreaking:
Jailbreaking refers to techniques used to bypass safety filters and alignment mechanisms in LLMs.
Common approaches include role-playing scenarios, token manipulation, and multi-step reasoning tricks
that gradually guide the model toward producing harmful content.

Data Poisoning:
Training data poisoning attacks involve injecting malicious examples into the training dataset to
introduce backdoors or biases. These attacks can be particularly effective because they occur before
deployment and may be difficult to detect through standard evaluation.

Defense Mechanisms:
Several defense strategies have been proposed including: input filtering and sanitization,
adversarial training, constitutional AI approaches, and output monitoring. Federated learning and
differential privacy techniques can also help protect against some categories of attacks.

Conclusion:
As LLMs become increasingly integrated into critical applications, addressing their security
vulnerabilities becomes paramount. Future work should focus on developing more robust evaluation
benchmarks and standardized defense frameworks.
""",
    },
]


def create_sample_docs():
    out_dir = Path("sample_docs")
    out_dir.mkdir(exist_ok=True)

    for paper in SAMPLE_PAPERS:
        path = out_dir / paper["filename"]
        path.write_text(paper["content"].strip(), encoding="utf-8")
        print(f"✅ Created: {path}")

    # Also create a metadata JSON for the demo
    import json
    meta = [
        {"filename": p["filename"], "title": p["title"], "url": p["url"], "type": p["type"]}
        for p in SAMPLE_PAPERS
    ]
    (out_dir / "papers_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n✅ Created: sample_docs/papers_meta.json")
    print(f"\n📚 {len(SAMPLE_PAPERS)} sample documents ready in ./sample_docs/")
    print("Note: These are .txt files. The app accepts PDFs — use real papers for production.")
    print("For the demo, you can rename these to .pdf extension and the pdfminer fallback will handle them.")


if __name__ == "__main__":
    create_sample_docs()
