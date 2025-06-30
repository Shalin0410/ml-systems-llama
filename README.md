# Final Project - EE 508: Hardware Foundations of Machine Learning, Spring 2025

This repository is intended as a minimal example to load Llama 3 models and run inference. It is based on the [official implementation](https://github.com/meta-llama/llama3) from Meta.
The following modifications have been made:

* Remove dependencies on the `fairscale` and `fire` packages.
* Remove the chat completion feature.
* Reorganize and simplify the code structure for the generation function. The `Generation` class is now the base class of the Llama model.
    

## Quick Start
1. Install required packages:

```
pip install -r requirements.txt
```

2. [Request access](https://www.llama.com/llama-downloads/) to Llama models and download the Llama3.2-1B model weights:
    * Install the Llama CLI in your preferred environment: `pip install llama-stack`.
    * Run `llama model list` to show the latest models available.
    * Run `llama model download --source meta --model-id Llama3.2-1B`.
    * When the script prompts for your unique custom URL, copy and paste the custom URL. (Clicking on the URL itself does not access the model):

3. Review and run the code `inference.py`:

```
python inference.py
```