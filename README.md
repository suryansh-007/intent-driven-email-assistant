# Intent-Driven Email Assistant

## Project Topic

Intent-Driven Email Categorization, Abstractive Summarization,
and Conditional Response Generation.

## Objective

The project proposes an intelligent email assistant that:

1. Classifies the intent of an incoming email.
2. Generates an abstractive summary of an email thread.
3. Generates a conditional response using the intent,
   summary, and persona information.

## Datasets

The project uses the following datasets:

- CSpace – email intent categorization
- Avocado – enterprise email intent classification
- EmailSum – email thread summarization
- Amazon Product-QA – persona/preference-conditioned generation

## Repository Structure

```text
intent-driven-email-assistant/
│
├── README.md
├── requirements.txt
│
├── src/
│   └── load_dataset.py
│
├── data/
│   └── README.md
│
└── notebooks/
    └── README.md
