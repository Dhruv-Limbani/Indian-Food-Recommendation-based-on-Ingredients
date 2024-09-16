import streamlit as st
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy

if 'ner' not in st.session_state:
    st.session_state['ner'] = spacy.load("final_model")

with open("resources.pkl","rb") as f:
    vectorizer,encoder,x_ingredients,x_categorical,x_features = pickle.load(f)

df = pd.read_csv("ext_igdts.csv",index_col=None)
def extract_ingredients(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.replace(","," ")
    text_list = [text.split()[i:i+512] for i in range(0,len(text.split()),512)]
    ingredients = []
    for x in text_list:
        x = " ".join(x)
        doc = st.session_state['ner'](x)
        ings = [ent.text for ent in doc.ents if ent.label_ == "FOOD"]
        ingredients.append(" ".join(ings))
    return " ".join(ingredients)

def recommend(text, num_recommendations=5):
    inp = extract_ingredients(text)
    ingredient_feat_vec = vectorizer.transform([inp])
    similarity_matrix = cosine_similarity(ingredient_feat_vec, x_ingredients)
    # ccd = np.array([[cuisine, course, diet]])
    # ccd_feat_vec = encoder.transform(ccd)
    # combined_feat_vec = sp.hstack([ingredient_feat_vec, ccd_feat_vec])
    similar_indices = similarity_matrix[0].argsort()[-num_recommendations:][::-1]
    return df.iloc[similar_indices]

st.header("Hello")

tx = st.text_input("Enter Ingredients: ")

if tx:
    st.dataframe(recommend(tx))