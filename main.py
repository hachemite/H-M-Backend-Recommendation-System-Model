import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import random
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sklearn.metrics.pairwise import euclidean_distances
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv  # <-- NEW

#ENV 
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")  # default fallback
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 120))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/hm_lean.db")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# PATHS
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "db" / "hm_lean.db"
IMAGE_DIR = BASE_DIR / "db" / "hm_lean_project" / "images_lean"
MODELS_DIR = BASE_DIR / "models"

# GLOBAL STATE Dictionary for ram managemest faster
brain = {}
prices_cache = {}

# database models user and etc 
class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    sex: Optional[str] = "Unknown"
    birth_date: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

# --- DB INIT ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            hashed_password TEXT,
            sex TEXT,
            birth_date TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cart (
            user_id INTEGER,
            article_id INTEGER,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, article_id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.close()

@asynccontextmanager  # trigger one time at startup/shutdown 
async def lifespan(app: FastAPI):
    print("Startup: Loading Brain & Cache...")
    init_db()
    
    # 1. Load AI Vectors
    if MODELS_DIR.exists():
        # pickle load the model to the ram faster pour io 
        with open(MODELS_DIR / "model_vectors.pkl", "rb") as f:
            data = pickle.load(f)
            brain['ids'] = data['article_ids'] if 'article_ids' in data else data['ids']
            brain['vectors'] = data['vectors']

 
    conn = sqlite3.connect(DB_PATH)
    db_ids = pd.read_sql("SELECT article_id FROM articles", conn)['article_id'].values
    
    print("Caching prices...")
    # On prend le MAX price pour être sûr (ou AVG)
    price_df = pd.read_sql("SELECT article_id, MAX(price) as price FROM transactions GROUP BY article_id", conn)
    global prices_cache
    prices_cache = dict(zip(price_df['article_id'], price_df['avg_price'] if 'avg_price' in price_df else price_df['price']))
    
    conn.close()

    # Filtre les vecteurs pour ne garder que ceux qui sont dans la DB
    mask = np.isin(brain['ids'], db_ids)
    brain['vectors'] = brain['vectors'][mask]
    brain['ids'] = brain['ids'][mask]
    
    print(f"Ready: {len(brain['ids'])} items active.")
    yield
    brain.clear()
    prices_cache.clear()

app = FastAPI(title="H&M ILIA Pro Shop", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, # Cross-Origin Resource Sharing
    allow_origins=["*"],
    allow_methods=["*"], # ALL methods are allowed get/post/put/delete
    allow_headers=["*"], # ALL headers are allowed let us use custom headers like Authorization
)
app.mount("/static", StaticFiles(directory=str(IMAGE_DIR)), name="static")

# AUTH 
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Validation failed")
    
    conn = sqlite3.connect(DB_PATH)
    user = conn.execute("SELECT id, username FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user is None: raise HTTPException(status_code=401, detail="User not found")
    return {"id": user[0], "username": user[1]}

# AUTH ENDPOINTS
@app.post("/register", status_code=201)
async def register(user: UserRegister):
    conn = sqlite3.connect(DB_PATH)
    try:
        hashed = pwd_context.hash(user.password)
        conn.execute("INSERT INTO users (username, email, hashed_password, sex, birth_date) VALUES (?,?,?,?,?)",
                     (user.username, user.email, hashed, user.sex, user.birth_date))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username taken")
    finally:
        conn.close()
    return {"msg": "User created"}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect(DB_PATH)
    user = conn.execute("SELECT hashed_password FROM users WHERE username = ?", (form_data.username,)).fetchone()
    conn.close()
    if not user or not pwd_context.verify(form_data.password, user[0]):
        raise HTTPException(status_code=401, detail="Bad credentials")
    return {"access_token": create_access_token({"sub": form_data.username}), "token_type": "bearer"}

# CART 
@app.post("/cart/add/{article_id}")
async def add_to_cart(article_id: int, user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO cart (user_id, article_id) VALUES (?, ?)", (user['id'], article_id))
        conn.commit()
    except sqlite3.IntegrityError:
        return {"msg": "Already in cart"}
    finally:
        conn.close()
    return {"msg": "Added"}

@app.delete("/cart/remove/{article_id}")
async def remove_from_cart(article_id: int, user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM cart WHERE user_id = ? AND article_id = ?", (user['id'], article_id))
    conn.commit()
    conn.close()
    return {"msg": "Removed"}

@app.get("/cart")
async def view_cart(user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT a.article_id, a.prod_name, a.product_type_name, a.detail_desc, a.colour_group_name 
        FROM articles a JOIN cart c ON a.article_id = c.article_id WHERE c.user_id = ?
    """
    df = pd.read_sql(query, conn, params=(user['id'],))
    conn.close()
    
    results = []
    for item in df.to_dict(orient="records"):
        item['price'] = prices_cache.get(item['article_id'], 0.0)
        item['image_path'] = f"/static/{str(item['article_id']).zfill(10)}.jpg"
        results.append(item)
    return results

# SHOP & AI ENDPOINTS pour boutique les element aleoratoire dans l'espace principale
def enrich_results(df_items):
    """Ajoute prix et image aux résultats"""
    results = []
    for item in df_items.to_dict(orient="records"):
        item['price'] = prices_cache.get(item['article_id'], 0.0)
        item['image_path'] = f"/static/{str(item['article_id']).zfill(10)}.jpg"
        # Nettoyage des valeurs nulles
        if 'detail_desc' in item and item['detail_desc'] is None:
            item['detail_desc'] = "No description available."
        results.append(item)
    return results

@app.get("/boutique")
async def get_boutique(product_group: str = None, max_price: float = None, limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT article_id, prod_name, product_type_name, product_group_name, detail_desc, colour_group_name, section_name FROM articles WHERE 1=1"
    params = []

    if product_group and product_group != "All":
        query += " AND product_group_name = ?"
        params.append(product_group)
    
    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit * 4) # Buffer pour filtrage python

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    results = []
    final_items = enrich_results(df)
    
    for item in final_items:
        if max_price and item['price'] > max_price: continue
        results.append(item)
        if len(results) >= limit: break

    return results

@app.get("/recommend/{article_id}")
async def recommend(article_id: int, top_k: int = 5):
    try:
        idx = np.where(brain['ids'] == article_id)[0][0]
        vec = brain['vectors'][idx].reshape(1, -1)
    except:
        raise HTTPException(404, "Item not in AI Model")

    dists = euclidean_distances(vec, brain['vectors']).flatten()
    indices = dists.argsort()[1:top_k+1] 
    rec_ids = brain['ids'][indices].tolist()

    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?']*len(rec_ids))
    query = f"SELECT article_id, prod_name, product_type_name, product_group_name, detail_desc, colour_group_name, section_name FROM articles WHERE article_id IN ({placeholders})"
    df = pd.read_sql(query, conn, params=rec_ids)
    conn.close()

    return {"source_item": article_id, "recommendations": enrich_results(df)}

@app.get("/categories")
async def get_categories():
    conn = sqlite3.connect(DB_PATH)
    cats = pd.read_sql("SELECT DISTINCT product_group_name FROM articles", conn)
    conn.close()
    return cats['product_group_name'].tolist()

@app.get("/random-id")
async def random_id():
    return {"article_id": int(random.choice(brain['ids']))}

@app.get("/article/{article_id}")
async def get_article(article_id: int):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT article_id, prod_name, product_type_name, product_group_name,
               detail_desc, colour_group_name, section_name
        FROM articles
        WHERE article_id = ?
    """
    df = pd.read_sql(query, conn, params=(article_id,))
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail="Article not found")

    item = df.iloc[0].to_dict()
    item['price'] = prices_cache.get(article_id, 0.0)
    item['image_path'] = f"/static/{str(article_id).zfill(10)}.jpg"
    return item

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)