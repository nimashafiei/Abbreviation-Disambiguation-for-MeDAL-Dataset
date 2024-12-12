# **Abbreviation Disambiguation for MeDAL Dataset**

![Python Version](https://img.shields.io/badge/python-3.x-blue)
![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## **Overview**

This project aims to **disambiguate abbreviations in medical text** by predicting the **full terms of abbreviations based on the given context**. Leveraging state-of-the-art **Natural Language Processing (NLP)** techniques and pretrained models, the system performs **token classification** to generate context-sensitive predictions for abbreviations.

---

## **Key Features**
- **Dataset**: Utilizes the MeDAL dataset with 73,196 preprocessed records.  
- **Abbreviation Disambiguation**: Handles **4,866 unique abbreviations** and their contextual variations.
- **Token Classification**: Maps abbreviations to their full forms with a token-labeling approach.
- **Pretrained Models**: Tested multiple models such as:
  - [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
  - [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)
  - [BlueBERT](https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12)
  - [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- **Metrics**: Evaluated with **F1-score, Precision, and Recall (macro & weighted)**.

---

## **Dataset**

### MeDAL Preprocessing:
1. Dataset initially contained **5M records**.
2. Filtered and preprocessed for efficient training:
   - Removed punctuation and stopwords.
   - Adjusted abbreviation locations for cleaned text.
3. **Final Dataset**:  
   - 73,196 records  
   - 300 most frequent abbreviations  
   - 1,005 corresponding labels  

---

## **Approach**

### Token Classification
- Every token in the text is classified as:
  - **Abbreviation**: Associated with the correct full form.
  - **Non-abbreviation**: Labeled as `NA_word`.

#### Example:

| **Original Text**              | employing cytochemical methods found early embryonic **OD** nuclei contain **CS** |
|---------------------------------|----------------------------------------------------------------------------------|
| **Predicted Labels**            | NA_word NA_word NA_word NA_word NA_word embryonic **optical density** nuclei contain **case series** |

---

## **Model Training**

- **Framework**: [HuggingFace Transformers](https://huggingface.co/transformers/)  
- **Training Details**:
  - Pretrained Model: `biobert-base-cased-v1.2`
  - Epochs: **4**
  - Batch Size: **8**
  - Learning Rate: `2e-5`

### Training Results
| **Epoch** | **Training Loss** | **Validation Loss** |
|-----------|--------------------|---------------------|
| 1         | 0.197             | 0.176              |
| 2         | 0.157             | 0.144              |
| 3         | 0.138             | 0.127              |
| 4         | 0.128             | 0.119              |

---

## **Evaluation**

The model was evaluated using **macro and weighted metrics**.  
- **Macro Metrics**:
  - F1-Score: `33.2%`
  - Precision: `38.27%`
  - Recall: `31.86%`

- **Weighted Metrics**:
  - F1-Score: `55.36%`
  - Precision: `65.97%`
  - Recall: `49.81%`

#### Insights:
- **Strengths**:
  - Model learned to predict labels for frequent abbreviations.
  - Significant improvement observed with additional epochs.
- **Areas of Improvement**:
  - Extend training to more epochs.
  - Increase dataset size for rare abbreviations.

---

## **How to Run**

### Prerequisites
1. Install Python (>= 3.7).  
2. Install required libraries:  
   ```bash
   pip install datasets evaluate transformers seqeval
## **Steps**

### Clone the repository:
```bash
git clone https://github.com/your-username/medal-abbreviation-disambiguation.git
cd medal-abbreviation-disambiguation

## Download the dataset and models.
### Train the model:
python train.py

###Evaluate the model:
python evaluate.py

## **Technologies Used**

- **Python**
- **HuggingFace Transformers**
- **SeqEval**
- **Matplotlib**

## **Future Work**

- Expand dataset coverage to include rare abbreviations.
- Experiment with additional pretrained medical models.
- Optimize hyperparameters for further performance improvements.
- Deploy the model using HuggingFace Inference API or Streamlit.

## **Contributing**
Contributions are welcome! Please create a pull request or open an issue to share ideas, bugs, or suggestions.



