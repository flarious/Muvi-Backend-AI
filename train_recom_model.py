from joblib import load

def getData():
    df = load("cache_data.joblib")
    return df

def create_cosine_sim_model():
    cos_sim_data = load("cache_cosine.joblib")
    return cos_sim_data

def train_model():
    df = getData()
    cos_sim_data = create_cosine_sim_model()

    return df, cos_sim_data