# Experiment 3: Text Generation — AI Question–Answer System

## Objective

To generate text and build an AI Question–Answer system using a pre-trained FLAN-T5 model.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- FLAN-T5 (`google/flan-t5-base`)
- Gradio
- Google Colab

## Features

- Enter questions in natural language.
- Generate answers using a pre-trained language model.
- Adjust the maximum answer length.
- Display generated answers in an interactive web interface.
- Clear the question and answer fields.
- Test questions related to Artificial Intelligence, Machine Learning, Deep Learning, and Python.

## How It Works

1. The user enters a question.
2. The question is converted into tokens.
3. The FLAN-T5 model processes the input.
4. The model generates an answer.
5. The generated answer is displayed in the Gradio interface.

**Workflow:**

User Question → Tokenization → FLAN-T5 Model → Text Generation → Generated Answer

## Source Code

The complete source code is available in the Jupyter Notebook:

`Text_Generation_AI_Question_Answer_System_FLAN_T5.ipynb`

## How to Run

1. Open the notebook in Google Colab.
2. Run the code cells in order.
3. Wait for the FLAN-T5 model to load.
4. Run the final Gradio interface cell.
5. Open the generated Gradio link.
6. Enter a question and click **Ask Question** to generate an answer.

## Example Questions

- What is artificial intelligence?
- What is machine learning?
- Explain deep learning in simple words.
- What is a neural network?
- What is generative AI?

## Live Demo
https://471714f143894a0ff6.gradio.live


