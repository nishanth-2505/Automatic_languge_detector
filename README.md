Automatic Language Detection using NLP
Project Description

This project is an Automatic Language Detection System built using Natural Language Processing (NLP) techniques.
It detects the language of a given text input and returns the predicted language along with a confidence score.

The project follows a structured NLP pipeline, including text cleaning, character n-gram generation, embedding creation, sentence vector computation, and language prediction.

Objective

To automatically identify the language of text entered by a user using NLP-based preprocessing and machine learning language detection models.

How the Project Works (Workflow)

The system processes input text in multiple sequential stages:

Step 1 — User Input

The user enters any sentence or phrase.

Example:

Hello how are you

Step 2 — Text Cleaning

The text is cleaned to remove:

Uppercase letters (converted to lowercase)

URLs

Special characters

Unwanted symbols

Purpose:

Reduce noise

Improve detection accuracy

Example:

HELLO!!! → hello

Step 3 — Character N-Gram Generation

The cleaned text is split into character groups of length 3.

Example:

hello → ['hel', 'ell', 'llo']


Why n-grams are used:

Capture language writing patterns

Identify differences between languages

Improve detection reliability

Step 4 — Embedding Generation

Each n-gram is converted into a numerical vector (embedding).

Embeddings represent:

Linguistic patterns

Semantic meaning

Structural information

Example embedding:

[0.23, 0.67, 0.91, 0.12, 0.45, 0.88]


Purpose:

Convert text into machine-readable numeric format

Step 5 — Sentence Vector Creation

All embedding vectors are averaged to create one final sentence-level vector.

Purpose:

Represent the entire sentence in one numeric form

Reduce data complexity

Enable efficient prediction

Step 6 — Language Detection

The processed text is passed to a language detection model (langdetect).

The model predicts:

Language name

Confidence probability

Example output:

English → Confidence: 0.99
French → Confidence: 0.01

Step 7 — Final Output

The system selects the highest-confidence prediction and displays:

Detected Language: English
Confidence Score: 0.9987

Project Pipeline Summary
User Input
   ↓
Text Cleaning
   ↓
Character N-Grams
   ↓
Embedding Generation
   ↓
Sentence Vector Creation
   ↓
Language Detection
   ↓
Final Output

Technologies Used

Python

Regular Expressions (re)

NumPy

Langdetect

Sentence Transformers (optional for embeddings)

Installation Steps
Step 1 — Create Virtual Environment
python -m venv ft-env

Step 2 — Activate Virtual Environment
ft-env\Scripts\activate

Step 3 — Install Required Packages
pip install langdetect numpy sentence-transformers

How to Run the Project
python detect.py


Then enter text when prompted.

Example Output
Enter text to detect language:
> Hello Baby

Detected Language: en
Confidence Score: 0.99

Real-World Applications

Google Translate language detection

Chatbots and virtual assistants

Social media content filtering

Multilingual customer support

Document classification

Voice assistants

Content moderation

Key Features

Fully automated NLP pipeline

Works with multiple languages

Modular structured code

Expandable to deep learning models

Beginner-friendly implementation

Future Improvements

Add FastText language model

Improve accuracy using Transformer-based models

Build a web interface using Flask or Streamlit

Add multilingual dataset training

Deploy as a REST API

Project Author

Developed by: Nishanth Thala

