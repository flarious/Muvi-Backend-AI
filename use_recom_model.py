from train_recom_model import train_model
import requests

df = None
cos_sim_data = None

def load_model():
    global df
    global cos_sim_data

    df, cos_sim_data = train_model()

def queried_df(query):
    # Get the queried movie either by title, genre, or overview
    return df[df['original_title'].str.contains(query) | df['genres'].str.contains(query) | df['lemmatized_overview'].str.contains(query)]

def get_movie_and_5_similar(query):
    # Recommendation using Cosine KNN
    selected_index = queried_df(query)
    if not selected_index.empty:
        cos_sim_data_query = cos_sim_data[selected_index.index[0]]
        index_recomm = cos_sim_data_query.index.tolist()
        return [list(x) for x in zip(index_recomm, cos_sim_data_query)]
    else:
        return []

def get_movie_name(index_recomm):
    # Show result in {"id": 155, "name": "the dark knight"} format
    movie_recomm = [{"id": df['movie_id'][i[0]].item(), "name": df['original_title'][i[0]]} for i in index_recomm]
    return movie_recomm

def give_recommendation(text):
    # Find similar words from query
    list_of_similar_words = []
    response = requests.get(f"http://127.0.0.1:8002/find_similar/{text}")
    if response.status_code == 200:
        list_of_similar_words = response.json()
        list_of_similar_words = list_of_similar_words["similar_words"]

    # Recommendation
    list_of_result = []
    for words in list_of_similar_words:
         for word in words:
            recommended_movie = get_movie_and_5_similar(word) 
            for movie in recommended_movie:
                # If first movie in the list, just add to result
                if list_of_result == []:
                    list_of_result.append(movie)
                # If not first movie, check if it duplicate or not. If duplicated, add cosine similarity score. If not, add movie to result
                else:
                    for result in list_of_result:
                        dup_movie_flag = 0
                        if movie[0] == result[0]:
                            result[1] += movie[1]
                            dup_movie_flag = 1
                            break
                    
                    if dup_movie_flag != 1:
                        list_of_result.append(movie)
        
    list_of_result = sorted(list_of_result, key = lambda t: t[1], reverse=True) # Sort result by cosine similarity score
    list_of_result = get_movie_name(list_of_result)

    return list_of_result
