from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import uvicorn,statistics
from pydantic import BaseModel
 
# Loading Summarisation Model
model_transformer = SentenceTransformer('kornwtp/ConGen-BERT-Tiny')

# Creating FastAPI instance
app = FastAPI()

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
        similarity_score = get_similarity_hybrid_model(data.text1,data.text2)
        jaccard_score = jaccard_similarity_score(data.text1,data.text2)
        levenshtein_score = levenshtein_similarity_score(data.text1,data.text2)
        # Return the Result
        scores = {"jaccard_score":jaccard_score,"levenshtein_score":levenshtein_score,"similarity score":similarity_score}
        normalized_scores = {key: (value + 1) / 2 if value < 0 else value for key, value in scores.items()}
        weights = {"jaccard_score":0.15,"levenshtein_score":0.15,"similarity score":0.7}
        cumulative_score = sum(normalized_scores[key] * weights[key] for key in normalized_scores.keys())
        return {"similarity Score":cumulative_score}
    except Exception as e:
        return {"similarity score":str(e)}