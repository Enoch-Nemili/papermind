# Retrieval eval


## plain English (26 questions)

| Method | hit@1 | recall@5 | MRR |
|---|---|---|---|
| semantic | 69% | 92% | 0.776 |
| keyword | 62% | 77% | 0.688 |
| hybrid w=0.5 | 69% | 96% | 0.779 |
| hybrid w=1.0 | 65% | 92% | 0.759 |
| hybrid w=1.5 | 65% | 88% | 0.758 |
| hybrid w=2.0 | 65% | 88% | 0.765 |
| hybrid w=3.0 | 73% | 88% | 0.804 |


Misses at #1, semantic (plain English):
- How does self-attention work?  ->  got t5.pdf
- Why is reading text in both directions better for understanding language?  ->  got gpt3-few-shot-learners.pdf
- Can learned embeddings beat classic keyword search at finding Wikipedia passages that answer a question?  ->  got bert.pdf
- Can a huge language model do a new task just from a few examples in the prompt, without retraining?  ->  got lora.pdf
- Can a smaller open model trained on more data match much bigger models?  ->  got t5.pdf
- What publicly available data was used to train an open foundation model?  ->  got gpt3-few-shot-learners.pdf
- Why can the changes made during fine-tuning be captured by small matrices?  ->  got t5.pdf
- How can a language model look up documents before it answers a question?  ->  got gpt3-few-shot-learners.pdf


Misses at #1, hybrid w=1.0 (plain English):
- How does self-attention work?  ->  got t5.pdf
- Why is reading text in both directions better for understanding language?  ->  got gpt3-few-shot-learners.pdf
- Can a huge language model do a new task just from a few examples in the prompt, without retraining?  ->  got chain-of-thought.pdf
- Can a smaller open model trained on more data match much bigger models?  ->  got t5.pdf
- What publicly available data was used to train an open foundation model?  ->  got t5.pdf
- How can I fine-tune a giant model cheaply without updating all of its weights?  ->  got t5.pdf
- Why can the changes made during fine-tuning be captured by small matrices?  ->  got t5.pdf
- How can a language model look up documents before it answers a question?  ->  got gpt3-few-shot-learners.pdf
- What is the Colossal Clean Crawled Corpus?  ->  got gpt3-few-shot-learners.pdf

## exact terms (10 questions)

| Method | hit@1 | recall@5 | MRR |
|---|---|---|---|
| semantic | 100% | 100% | 1.000 |
| keyword | 80% | 90% | 0.831 |
| hybrid w=0.5 | 100% | 100% | 1.000 |
| hybrid w=1.0 | 100% | 100% | 1.000 |
| hybrid w=1.5 | 90% | 100% | 0.950 |
| hybrid w=2.0 | 90% | 100% | 0.950 |
| hybrid w=3.0 | 90% | 100% | 0.925 |


Misses at #1, semantic (exact terms):
- none


Misses at #1, hybrid w=1.0 (exact terms):
- none

## combined (36 questions)

| Method | hit@1 | recall@5 | MRR |
|---|---|---|---|
| semantic | 78% | 94% | 0.838 |
| keyword | 67% | 81% | 0.728 |
| hybrid w=0.5 | 78% | 97% | 0.841 |
| hybrid w=1.0 | 75% | 94% | 0.826 |
| hybrid w=1.5 | 72% | 92% | 0.811 |
| hybrid w=2.0 | 72% | 92% | 0.816 |
| hybrid w=3.0 | 78% | 92% | 0.837 |


**Decision rule winner: hybrid w=0.5**
