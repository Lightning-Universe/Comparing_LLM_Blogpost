# Comparing Different Language Models
This repository contains the code used to create the results presented in the [blog post](https://www.notion.so/lightningai/Comparing-different-LLMS-Blogpost-4744d830674244dc90b38a8857da97d5?d=ea1cdaf3ddeb4fd59f17acab948629b7#3045ef91ef7d475982010f6f7524fa49) titled "Comparing different LLMs".

## Overview
The `models_HF.py` file wraps different Language Models (LLMs) available in the transformers library inside the `LLM` module from `langchain`. On the other hand, the `model_lit_llama.py` file wraps Lit-LLaMA inside the same module. Note that to use Lit-LLaMA, you'll need to request the weights from [Meta](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).

In the `comparing_LLM.ipynb` notebook, you'll be able to call any of the models seen in the blog (GPT4 and Bloomz 176b) and test its performance using the template.

## Requirements
To run the code, you need to have the following requirements:

Python 3
Jupyter Notebook

## How to use
Clone the repository to your local machine.
Install the required dependencies: `pip install -r requirements.txt`
Open `comparing_LLM.ipynb` using Jupyter Notebook.
Follow the instructions in the notebook to test the different LLMs.


