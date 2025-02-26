# 📧 Email/SMS Spam Classifier  
A **machine learning-based web app** built using **Streamlit, NLTK, and Scikit-learn** to classify messages as **Spam** or **Not Spam**. The model is trained using **TF-IDF vectorization** and a **Naive Bayes classifier** for high accuracy.  

🔗 **Live Demo:** [Try the App](https://vish-email-spam-classifier.streamlit.app)  

## Features

- **User-Friendly Interface:** A clean and interactive Streamlit interface where users can input messages and receive predictions in real-time.
- **Text Preprocessing:** Utilizes NLP techniques to clean and prepare text data by:
  - Converting text to lowercase.
  - Tokenizing sentences.
  - Removing non-alphanumeric tokens.
  - Filtering out stopwords and punctuation.
  - Applying Porter stemming to reduce words to their base forms.
- **Machine Learning Prediction:** 
  - Transforms preprocessed text using a TF-IDF vectorizer.
  - Uses a pre-trained classification model to predict whether the message is spam or not.

## How It Works

1. **Preprocessing:**  
   The `transform_text` function processes the input message by:
   - Lowercasing and tokenizing the text.
   - Removing non-alphanumeric tokens.
   - Eliminating stopwords and punctuation.
   - Applying Porter Stemming to simplify words.

2. **Vectorization:**  
   The preprocessed text is converted into numerical features using a TF-IDF vectorizer loaded from `vectorizer.pkl`.

3. **Prediction:**  
   The transformed text is passed to a machine learning model (loaded from `model.pkl`) which outputs a prediction:
   - **1** indicates that the message is **Spam**.
   - **0** indicates that the message is **Not Spam**.

4. **Display:**  
   The result is shown on the Streamlit app interface, informing the user whether the input message is spam or not.
 

## 🏗️ Installation  
```bash
git clone https://github.com/Viishwaajeet/email-spam-classifier.git
cd email-spam-classifier
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
streamlit run app.py
```

# Model and Vectorizer Files

Ensure that the following pickle files are in the repository’s root directory:

- **vectorizer.pkl** (TF-IDF vectorizer)
- **model.pkl** (pre-trained spam classification model)

# Future Enhancements

- Incorporate additional NLP techniques for improved preprocessing.
- Experiment with various machine learning models to enhance classification accuracy.
- Deploy the application on a cloud platform for easy accessibility.
