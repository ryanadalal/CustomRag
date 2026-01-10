Project: Hallucination Detection & Retrieval Failure Analysis for Sports RAG
High-level goal

Build a RAG system for sports facts and a diagnostic layer that detects when and why the model hallucinates.

Key idea:

Sports are fact-heavy, time-indexed, and easy to verify

Perfect for grounding + failure analysis

1️⃣ Domain & Data (Very Important)
Choose a narrow sports domain

Examples:

NBA games (last 10 seasons)

Soccer matches in one league

MLB player stats

Tennis Grand Slam results

Why narrow?

Clear ground truth

Easier eval

Cleaner hallucination signals

Data sources

Structured data: box scores, tables

Semi-structured: recaps, articles

You want both, so retrieval can fail in different ways.

2️⃣ Baseline RAG System (No Frameworks)

You build this from scratch.

Pipeline

Chunking

By game

By player-season

Embedding

Vector search

Context selection

Answer generation

Queries to support

“Who scored the most points in the 2019 NBA Finals?”

“Did LeBron play in Game 6 of the 2016 Finals?”

“What was the final score of X vs Y?”

3️⃣ Hallucination Taxonomy (Critical & Rare Skill)

You explicitly define hallucinations:

Type	Example
Fabricated fact	Player played when they didn’t
Partial truth	Correct player, wrong stat
Temporal error	Using data from wrong season
Retrieval omission	Info exists but not retrieved
Overgeneralization	Model fills gaps

This alone is very impressive.

4️⃣ Hallucination Detection Methods

You implement multiple detectors and compare them.

A. Retrieval-Grounded Verification

Extract claims from output

Check if claims appear in retrieved context

Flag unsupported claims

B. Self-Consistency Checks

Re-ask question with:

different wording

different retrieval seeds

Measure variance

High variance → likely hallucination

C. Answer-Context Entailment

Run a smaller NLI-style model

Check if context ⇒ answer

This is very technical.

D. “Abstain-or-Answer” Classifier

Train a small classifier:

Input: question + retrieved context

Output: answerable / not answerable

If not answerable → force refusal

5️⃣ Retrieval Failure Analysis Tool (This Is the Gold)

You now analyze why hallucinations happened.

Failure categories
Failure	Cause
Missed chunk	Bad chunking
Wrong chunk	Embedding similarity failure
Truncated context	Context window overflow
Query mismatch	Poor query reformulation
Tool outputs

For each query:

Retrieved chunks (ranked)

Missing gold chunk

Token budget usage

Final answer

You are essentially building a debugger for RAG.

6️⃣ Experiments You Can Run (Research-Grade)
Experiment 1: Chunking Strategy

By game vs by paragraph