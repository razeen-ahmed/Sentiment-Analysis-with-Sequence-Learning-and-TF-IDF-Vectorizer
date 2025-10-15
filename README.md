# Sentiment-Analysis-with-Sequence-Learning-and-TF-IDF-Vectorizer

---

## 📘 Overview  

This repository implements **sentiment analysis** on IMDB movie reviews as part of the *Natural Language Preprocessing-1* coursework.  
The project demonstrates the full progression from **classical text classification** methods to **deep learning sequence models** using **pretrained GloVe embeddings**, analyzing the impact of feature representation and model architecture on performance.

It provides a comparative perspective between **TF-IDF-based linear models** and **neural sequence learning architectures**, emphasizing the tradeoff between expressivity and trainability.

---

## 🎯 Objectives  

1. **Preprocessing and Vectorization:**  
   - Clean and tokenize raw text data.  
   - Represent text using **Bag-of-Words (CountVectorizer)** and **TF-IDF**.  

2. **Classical Machine Learning Models:**  
   - Implement **Logistic Regression** and **Naive Bayes** classifiers for baseline performance.  

3. **Deep Learning Sequence Model:**  
   - Construct a **Keras Sequential model** using **pretrained GloVe embeddings (100D)**.  
   - Include an **Embedding layer**, **Dense hidden layer**, and **Output layer** for binary classification.  

4. **Performance Comparison and Analysis:**  
   - Compare training efficiency and accuracy between traditional and neural approaches.  
   - Analyze how sequence representation (word embeddings) influences sentiment understanding.  

---

## 🧩 Workflow Pipeline  

### **1. Data Loading**
- IMDB movie reviews dataset is loaded using Pandas.  

### **2. Preprocessing**
- Text normalization: lowercasing, tokenization, and removal of stopwords.  
- Conversion into feature vectors via:  
  - **CountVectorizer (BoW)**  
  - **TF-IDF Vectorizer**  

### **3. Model Training**
- **Classical Models:** Logistic Regression, Naive Bayes.  
- **Neural Model:** Sequential network using pretrained GloVe embeddings.  

### **4. Evaluation & Visualization**
- Accuracy and loss curves plotted using Matplotlib.  
- Model comparison table generated to illustrate relative expressivity.  

---

## ⚙️ Technology Stack  

| Category | Tools/Libraries |
|-----------|-----------------|
| Programming Language | Python 3.10+ |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| NLP Utilities | GloVe (100D embeddings) |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

## 🧠 Implementation Details  

### **Feature Extraction**
- **TF-IDF Vectorizer**: Highlights important terms using inverse document frequency.  
- **CountVectorizer (BoW)**: Serves as a simpler baseline model.  

### **Model Architectures**
1. **Logistic Regression / Naive Bayes** – Fast and interpretable; limited contextual awareness.  
2. **Sequential Neural Network**  
   - Embedding Layer (GloVe 100D pretrained vectors)  
   - Dense hidden layer with ReLU activation  
   - Output layer (sigmoid) for binary sentiment classification  

### **Training Parameters**
- Optimizer: Adam  
- Loss: Binary Cross-Entropy  
- Metrics: Accuracy  
- Validation Split: 0.2  

---

## 📊 Results Summary  

| Model | Vectorizer | Accuracy | Notes |
|--------|-------------|-----------|--------|
| Logistic Regression | Bag-of-Words | ~85% | Strong baseline model |
| Naive Bayes | TF-IDF | ~83% | Lightweight, interpretable |
| Sequential NN | GloVe Embedding | ~88–90% | Context-aware, expressive |

> The GloVe-based model achieved the highest accuracy due to its ability to encode semantic relationships.

---

## 🧩 Repository Structure  

```
sentiment-sequence-learning-tfidf-glove/
│
├── Sentiment-Analysis-with-SeqLearning.ipynb        # Jupyter notebook with full implementation
├── README.md               # Documentation (this file)
└── requirements.txt        # Optional dependency list
```

---

## 🔬 Key Takeaways  

- **Expressivity:** Neural models capture deeper semantic meaning due to embeddings.  
- **Trainability:** Simpler models train faster but lack contextual generalization.  
- **Tradeoff:** The project highlights how feature representation impacts learning performance — a key insight in sequence modeling.  

---

## 🧪 Reproducibility  

### Installation  
```bash
git clone https://github.com/<your-username>/sentiment-sequence-learning-tfidf-glove.git
cd sentiment-sequence-learning-tfidf-glove
pip install -r requirements.txt
```


### Requirements  
```bash
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
```

Optional for embeddings:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

---

## 🧩 Future Work  

- Extend to **LSTM / BiLSTM** for capturing long-range dependencies.  
- Incorporate **attention mechanisms** for improved interpretability.  
- Explore **transfer learning** using BERT or DistilBERT for enhanced contextual performance.

---

## ✍️ Author  

**Razeen Ahmed**  
Department of Computer Science & Engineering  
BRAC University  
📧 *ahmedshadman12@gmail.com*  

---

## 📜 License  

MIT License © 2025 Razeen Ahmed  
