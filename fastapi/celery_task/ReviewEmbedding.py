import requests
import json
import numpy as np
from requests.api import post
import transformers
import time
import os
import faiss
import re

url = 'http://localhost:8501/v1/models/rest_review_distilbert:predict'


class RestReview:
    """ Wrapper for loading and serving yolov5s model"""
    
    def __init__(self, tokenizer_path, rest_data_root):
        self.max_length = 512
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        # Load restaurant embeddings
        filepaths = self.get_filepaths(rest_data_root)
        ## Holds all restaurant info
        rest_ratings = []  # store ratings, get sorted index, rearrange info and embeddings by rating
        self.restaurant_info = []
        rest_postal_ratings = {}
        self.rest_postal_info = {}
        rest_region_ratings = {}
        self.rest_region_info = {}
        ## Holds resturant info for postal code. Key: first 3 digit of postal code. Item: emb/info
        restaurant_embeddings = []
        rest_postal_embeddings = {}
        rest_region_embeddings = {}
        for idx, rest_path in enumerate(filepaths):
            with open(rest_path,'r') as f:
                # Get data for this restaurant
                rest_data = json.load(f)
                if 'Singapore' not in rest_data['address']:
                    continue
                # Get embedding
                restaurant_embeddings.append(rest_data['embedding'])
                # Get info to match embedding (based on index)
                rest_info = {
                            "name": rest_data["name"],
                            "address": rest_data["address"],
                            "tags": rest_data["review_tags"],
                            "link": rest_data["url"],
                            "about": rest_data["about"],
                            "summary": rest_data["summary"],
                            "rating": rest_data["rating"]
                        }
                rest_ratings.append(rest_data['rating'])
                self.restaurant_info.append(rest_info)

                # Get info for postal code matching
                postal_code = re.findall("(Singapore) ([0-9]{6,6})", rest_data["address"])
                if not postal_code:
                    continue
                postal_code = postal_code[0][1][:2] # get first 2 digit
                if self.rest_postal_info.get(postal_code, -1) == -1:
                    self.rest_postal_info[postal_code] = [rest_info]
                    rest_postal_ratings[postal_code] = [rest_data['rating']]
                else:
                    self.rest_postal_info[postal_code].append(rest_info)
                    rest_postal_ratings[postal_code].append(rest_data['rating'])
                # Get embeddings for postal code matching
                if rest_postal_embeddings.get(postal_code, -1) == -1:
                    rest_postal_embeddings[postal_code] = [rest_data['embedding']]
                else:
                    rest_postal_embeddings[postal_code].append(rest_data['embedding'])

                # Get info for region matching
                region = rest_data['region']
                if self.rest_region_info.get(region, -1) == -1:
                    self.rest_region_info[region] = [rest_info]
                    rest_region_ratings[region] = [rest_data['rating']]
                else:
                    self.rest_region_info[region].append(rest_info)
                    rest_region_ratings[region].append(rest_data['rating'])

                # Get embeddings for region matching
                if rest_region_embeddings.get(region, -1) == -1:
                    rest_region_embeddings[region] = [rest_data['embedding']]
                else:
                    rest_region_embeddings[region].append(rest_data['embedding'])

                
        # Sort
        sorted_idx = np.argsort(-np.asarray(rest_ratings))
        self.restaurant_info = np.asarray(self.restaurant_info)[sorted_idx]
        restaurant_embeddings = np.asarray(restaurant_embeddings)[sorted_idx]
        # Create FAISS index
        restaurant_embeddings = restaurant_embeddings.astype('float32')
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.index.add_with_ids(restaurant_embeddings, np.array(range(0,len(restaurant_embeddings))).astype('int64'))
        faiss.write_index(self.index, 'rest_emb.index')

        # Create FAISS index for postal code search
        self.postal_index = {}
        for postal_code, embeddings in rest_postal_embeddings.items():
            # Sort
            sorted_idx = np.argsort(-np.asarray(rest_postal_ratings[postal_code]))
            self.rest_postal_info[postal_code] = np.asarray(self.rest_postal_info[postal_code])[sorted_idx]
            embeddings = np.asarray(embeddings)[sorted_idx]
            # Get index
            rest_embeddings = embeddings.astype('float32')
            self.postal_index[postal_code] = faiss.IndexIDMap(faiss.IndexFlatIP(768))
            self.postal_index[postal_code].add_with_ids(rest_embeddings, np.array(range(0,len(rest_embeddings))).astype('int64'))
            faiss.write_index(self.postal_index[postal_code], 'rest_postal_emb.index')
        
        # Create FAISS index for region code search
        self.region_index = {}
        for region, embeddings in rest_region_embeddings.items():
            # Sort
            sorted_idx = np.argsort(-np.asarray(rest_region_ratings[region]))
            self.rest_region_info[region] = np.asarray(self.rest_region_info[region])[sorted_idx]
            embeddings = np.asarray(embeddings)[sorted_idx]
            # Get index
            rest_embeddings = embeddings.astype('float32')
            self.region_index[region] = faiss.IndexIDMap(faiss.IndexFlatIP(768))
            self.region_index[region].add_with_ids(rest_embeddings, np.array(range(0,len(rest_embeddings))).astype('int64'))
            faiss.write_index(self.region_index[region], 'rest_postal_emb.index')
            
        print('Loaded model')

    @staticmethod
    def get_filepaths(root):
        for _root, dirs, files in os.walk(root):
            return [os.path.join(root, name) for name in files]

    # Get embedding from tfserving model
    # Returns embedding, err
    def get_emb(self, text, url):
        # Tokenize text
        model_input = self.tokenizer.encode_plus(text, truncation=True, max_length=self.max_length)
        model_input = {name : np.int64(value).tolist() for name, value in model_input.items()}
        
        # Prepare request
        headers = {"content-type": "application/json"}
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": [model_input]
        })
        try:
            json_response = requests.post(url, data=data, headers=headers)
        except Exception as e:
            return None, type(e).__name__

        embedding = json.loads(json_response.text)['predictions'][0]
        embedding = np.asarray(embedding)
        
        return embedding, None
    
    '''
    Search faiss
    query = query
    top_k = number of results
    index = faiss index
    model = embedding model
    '''
    def search(self, query_vector, top_k):
        top_k = min(top_k, len(self.restaurant_info))
        top_k = self.index.search(np.array([query_vector]), top_k)
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results =  [self.restaurant_info[idx] for idx in top_k_ids]
        return results

    '''
    Return results, time_taken, error string
    '''
    def predict(self, text, top_k, url):
        t=time.time()
        embedding, err = self.get_emb(text, url)
        if err:
            return [], 0, err
        embedding = np.array(embedding).astype(np.float32)     
        results = self.search(embedding, top_k=top_k)
        time_taken = time.time() - t
        return results, time_taken, ''

    '''
    Similar search, but restrict to postal code
    '''
    def search_postal(self, query_vector, top_k, postal_code):
        top_k = min(top_k, len(self.rest_postal_info[postal_code]))
        if len(self.rest_postal_info.get(postal_code, [])) == 0:
            return []
        top_k = self.postal_index[postal_code].search(np.array([query_vector]), top_k)
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results =  [self.rest_postal_info[postal_code][idx] for idx in top_k_ids]
        return results

    def predict_postal(self, text, top_k, postal_code, url):
        t=time.time()
        embedding, err = self.get_emb(text, url)
        if err:
            return [], 0, err 
        embedding = np.array(embedding).astype(np.float32)     
        results = self.search_postal(embedding, top_k=top_k, postal_code=postal_code)
        time_taken = time.time() - t
        return results, time_taken, ''
    
    '''
    Similar search, but restrict to region
    '''
    def search_region(self, query_vector, top_k, region):
        top_k = min(top_k, len(self.rest_region_info[region]))
        if len(self.rest_region_info.get(region, [])) == 0:
            return []
        top_k = self.region_index[region].search(np.array([query_vector]), top_k)
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results =  [self.rest_region_info[region][idx] for idx in top_k_ids]
        return results

    def predict_region(self, text, top_k, region, url):
        t=time.time()
        embedding, err = self.get_emb(text, url)
        if err:
            return [], 0, err  
        embedding = np.array(embedding).astype(np.float32)     
        results = self.search_region(embedding, top_k=top_k, region=region)
        time_taken = time.time() - t
        return results, time_taken, ''

    '''
    Random sample k restaurants
    '''
    def search_random(self, k):
        t=time.time()
        k = min(k, len(self.restaurant_info))
        results = np.random.choice(self.restaurant_info, k, replace=False).tolist()
        time_taken = time.time() - t
        return results, time_taken

    '''
    Random sample k restaurants in area
    '''
    def search_postal_random(self, k, postal_code):
        if len(self.rest_postal_info.get(postal_code, [])) == 0:
            return [], 0
        t=time.time()
        k = min(k, len(self.rest_postal_info[postal_code]))
        results = np.random.choice(self.rest_postal_info[postal_code], k, replace=False).tolist()
        time_taken = time.time() - t
        return results, time_taken

    '''
    Random sample k restaurants in region
    '''
    def search_region_random(self, k, region):
        if len(self.rest_region_info.get(region, [])) == 0:
            return [], 0
        t=time.time()
        k = min(k, len(self.rest_region_info[region]))
        results = np.random.choice(self.rest_region_info[region], k, replace=False).tolist()
        time_taken = time.time() - t
        return results, time_taken

    '''
    Get top k restaurants
    '''
    def search_topk(self, k):
        t=time.time()
        k = min(k, len(self.restaurant_info))
        results = self.restaurant_info[:k].tolist()
        time_taken = time.time() - t
        return results, time_taken

    '''
    Get top k restaurants in area
    '''
    def search_postal_topk(self, k, postal_code):
        if len(self.rest_postal_info.get(postal_code, [])) == 0:
            return [], 0
        t=time.time()
        k = min(k, len(self.rest_postal_info[postal_code]))
        results = self.rest_postal_info[postal_code][:k].tolist()
        time_taken = time.time() - t
        return results, time_taken

    '''
    Get top k restaurants in region
    '''
    def search_region_topk(self, k, region):
        if len(self.rest_region_info.get(region, [])) == 0:
            return [], 0
        t=time.time()
        k = min(k, len(self.rest_region_info[region]))
        results = self.rest_region_info[region][:k].tolist()
        time_taken = time.time() - t
        return results, time_taken

if __name__ == "__main__":
    tokenizer_path = "./token/review_emb-msmarco-distilbert-base-v3-v1"
    rest_data_path = "./restaurant_data"
    model = RestReview(tokenizer_path, rest_data_path)
    model.search_postal_random(5, '23')