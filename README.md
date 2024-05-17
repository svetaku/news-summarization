## Text Summarization with Transformers
This repository contains a Jupyter Notebook, <code>summarization-with-mt5.ipynb</code>, which demonstrates text summarization using the Transformer architecture. The notebook contains all the code necessary for training the model, evaluating its performance, and generating summaries.

### Dataset
The text summarization model is trained on the [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) dataset obtained from Kaggle. The dataset includes news articles and corresponding headlines. The 'ctext' column contains complete news article texts, while the 'headlines' column serves as reference summaries.

### Model and Training
The model architecture used for text summarization is Google's [MT5-small](https://huggingface.co/google/mt5-small) model, implemented using the AutoModelForSeq2SeqLM class from the Transformers library. Training is performed using the Trainer API with Seq2SeqTrainingArguments and Seq2SeqTrainer classes. [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) (Recall-Oriented Understudy for Gisting Evaluation) scores are used as evaluation metrics to monitor the model's performance.

### Baseline and Evaluation
As a baseline, the first three sentences of each article are used as summaries, and ROUGE scores are calculated between these baselines and the reference headlines. The notebook showcases the improvement achieved by the trained model compared to this baseline.

### Trained Model
The trained text summarization model is available in [this repository](https://huggingface.co/svetaku/mt5-small-finetuned-news-summary-kaggle).

## Usage
- Clone the repository
  ```sh
  git clone https://github.com/svetaku/news-summarization.git
  cd news-summarization
- Install dependencies
  ```sh
  pip install -r requirements.txt
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sunnysai12345/news-summary) or use your own
- Run *summarization-with-mt5.ipynb* notebook using your Hugging Face Hub access token when requested
