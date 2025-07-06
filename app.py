import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps= PorterStemmer()


def transform_text(text):
    text= text.lower()
    text= nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text =y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("📩 SMS or Email Spam Classifier")
input_sms = st.text_area("Enter multiple messages (one per line)")

# # 1. preprocess
# transformed_sms= transform_text(input_sms)
# # 2. vectorize
# vector_input= tfidf.transform([transformed_sms])
# # 3. Predict
# result= model.predict(vector_input)[0]
# # 4. Display
# if result ==1:
#     st.header("spam")
# else:
#     st.header("Not Spam")


# if st.button("Predict"):
#     if input_sms.strip() == "":
#         st.warning("⚠️ Please enter a message before predicting.")
#     else:
#         # 1. Preprocess
#         transformed_sms = transform_text(input_sms)

#         # 2. Vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # 3. Predict
#         result = model.predict(vector_input)[0]

#         # 4. Display
#         if result == 1:
#             st.header(" Spam")
#         else:
#             st.header(" Not Spam")

if st.button("Predict"):
    messages = input_sms.strip().split("\n")
    if len(messages) == 0 or messages == [""]:
        st.warning("⚠️ Please enter at least one message.")
    else:
        for i, msg in enumerate(messages):
            transformed = transform_text(msg)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]

            st.markdown(f"**Message {i+1}:** {msg}")
            if result == 1:
                st.error("→ 🚫 Spam")
            else:
                st.success("→ ✅ Not Spam")
