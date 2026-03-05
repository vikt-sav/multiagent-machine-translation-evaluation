# Multi-Agent Machine Translation Evaluation

A research prototype for evaluating machine translation using a multi-agent LLM system.

The system simulates a panel of AI agents that analyze translation quality through discussion and critique.

## Motivation

Traditional MT evaluation metrics (BLEU, TER, etc.) rely on reference similarity and often fail to capture semantic adequacy and translation errors.

Recent work shows that large language models can provide more human-aligned evaluation signals.

This project explores whether **multiple interacting agents** can improve evaluation reliability through discussion and disagreement resolution.

## Approach

The system uses several specialized agents:

1. **Translation Agent**
   - generates the candidate translation

2. **Critic Agent**
   - identifies potential translation errors

3. **Reviewer Agent**
   - evaluates the critic's claims

4. **Judge Agent**
   - produces the final evaluation score

The agents interact through structured prompts and iterative discussion.

## Pipeline

1. Source sentence
2. Candidate translation
3. Multi-agent discussion
4. Error identification
5. Final evaluation score
