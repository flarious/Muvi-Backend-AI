from fastapi import FastAPI
from threading import Thread
from use_recom_model import give_recommendation, load_model

app = FastAPI()

@app.get("/")
async def placeholder():
    return {"status": "This is placeholder. Loading model"}

# search movie using keyword
@app.get("/search/{keyword}")
async def search_keyword(keyword: str):
    movies = give_recommendation(keyword)
    return {"movies": movies}

# Workaround in case of loading cache take too long
def load_model_at_startup():
    print("Model is loading")

    load_model()

    print("Model has been loaded")

Thread(target=load_model_at_startup).start()