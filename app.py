from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import uvicorn,statistics
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('stopwords')

 
# Loading Summarisation Model
model_transformer = SentenceTransformer('mrp/SCT_Distillation_BERT_Tiny')
tokenizer_summarisation = AutoTokenizer.from_pretrained("slauw87/bart_summarisation")
summarizer_summarisation = AutoModelForSeq2SeqLM.from_pretrained("slauw87/bart_summarisation")


# Creating FastAPI instance
app = FastAPI()

def summarize_text(text):
    global tokenizer_summarisation,summarizer_summarisation
    inputs = tokenizer_summarisation(text,return_tensors='pt')['input_ids']
    summary_ids = summarizer_summarisation.generate(inputs, max_length=250,length_penalty=3.0,num_beams=2)
    summary =  tokenizer_summarisation.decode(summary_ids[0] , skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary

def tfidf_similarity_score(text1, text2):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0,0]
    
    return float(similarity)

def jaccard_similarity_score(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union

def levenshtein_similarity_score(text1, text2):
    seq_matcher = SequenceMatcher(None, text1, text2)
    return seq_matcher.ratio()

def get_similarity_hybrid_model(text1_summary,text2_summary):
    global model_transformer
    sentences = [text1_summary,text2_summary]
    embeddings = model_transformer.encode(sentences)
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0, 0]
    if float(similarity_score)< -1:
        return -1
    elif float(similarity_score)> 1:
        return 1
    else:
        return float(similarity_score)

# Creating class to define the request body
class request_body(BaseModel):
    text1 : str
    text2 : str
 
 
# Creating an Endpoint to receive the data to make prediction on.
@app.post('/predict')
def predict(data : request_body):
    try:
        if len(data.text1) >=250 or len(data.text2) >=250:
            text1 = summarize_text(data.text1)
            text2 = summarize_text(data.text2)
        else:
            text1 = data.text1
            text2 = data.text2

        similarity_score = get_similarity_hybrid_model(text1,text2)
        jaccard_score = jaccard_similarity_score(text1,text2)
        levenshtein_score = levenshtein_similarity_score(text1,text2)
        tfidf_score = tfidf_similarity_score(text1,text2)

        # Return the Result
        scores = {"jaccard_score":jaccard_score,"levenshtein_score":levenshtein_score,
                    "tfidf_score":tfidf_score,"similarity score":similarity_score}
        normalized_scores = {key: (value + 1) / 2 if value < 0 else value for key, value in scores.items()}
        weights = {"jaccard_score":0.1,"levenshtein_score":0.1,
                    "tfidf_score":0.1,"similarity score":0.7}
        
        cumulative_score = sum(normalized_scores[key] * weights[key] for key in normalized_scores.keys())
        return {"similarity Score":cumulative_score}
    except Exception as e:
        return {"similarity score":"Error Occured"}