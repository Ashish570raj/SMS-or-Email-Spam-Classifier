# 📧 SMS/Email Spam Classifier

A machine learning-based application to detect spam messages from SMS or emails using Natural Language Processing (NLP). Built with Python, trained on 5.5k+ labeled messages, and deployed using Streamlit for real-time predictions.

![Streamlit UI Screenshot](screenshot.png) <!-- Optional: Add your app screenshot here -->

## 🔍 Features
- Text preprocessing (lowercasing, stopword removal, stemming)
- TF-IDF vectorization for feature extraction
- Model training using Multinomial Naive Bayes, Random Forest, Logistic Regression, and XGBoost
- Soft voting ensemble for final prediction
- Deployed with a user-friendly Streamlit interface
- High accuracy: **97.8%**, with **100% precision** on spam detection

<!--
## 🚀 Demo
Click here to run the project locally, or [check the deployed version](#)  Add Streamlit share link if deployed -->

## 📁 Dataset
Used the popular [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) with 5,500+ messages classified as `ham` or `spam`.

## ⚙️ How It Works
1. **Preprocessing:** Clean and transform text (tokenize, remove stopwords, stem)
2. **Vectorization:** Convert text to numerical format using TF-IDF
3. **Model Training:** Compare models using accuracy and precision
4. **Voting Classifier:** Use ensemble of NB, RF, and XGB for best results
5. **Deployment:** Built an interactive UI using Streamlit

## 📦 Installation

```bash
git clone https://github.com/Ashish570raj/SMS-or-Email-Spam-Classifier.git
cd SMS-or-Email-Spam-Classifier
pip install -r requirements.txt
