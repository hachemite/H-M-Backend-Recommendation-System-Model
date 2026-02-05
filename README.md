# Shop♡Ease Backend: AI Recommendation System

This is the Python-based core of the Shop♡Ease platform. It manages user authentication, product management, and an AI engine that provides real-time fashion recommendations using Euclidean distance on pre-computed image/text vectors.

## 🧠 AI & Data Features
- **Limited Dataset:** Optimized for performance with a lean set of 6,520 articles.
- **Brain Engine:** Loads pre-trained vectors into RAM on startup for sub-millisecond recommendation latency.
- **Euclidean Similarity:** Finds the "closest" fashion items based on visual and categorical features.

---

## 🛠 Tech Stack
- **Framework:** FastAPI
- **Database:** SQLite (SQLAlchemy/Pandas for queries)
- **AI/ML:** Scikit-Learn (Euclidean Distances), NumPy, Pickle
- **Security:** JWT (JSON Web Tokens), Passlib (BCrypt hashing)

---

## 📂 Project Structure
Ensure your directory looks like this for the code to run correctly:
```text
.
├── main.py              # Application entry point
├── .env                 # Secret keys and config
├── db/
│   ├── hm_lean.db       # SQLite Database
│   └── hm_lean_project/
│       └── images_lean/ # Product .jpg images
└── models/
    └── model_vectors.pkl # Pre-trained AI vectors

```

---

## 🚀 Installation & Setup

1. **Create a Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


2. **Install Dependencies:**
```bash
pip install fastapi uvicorn scikit-learn pandas numpy python-jose[cryptography] passlib[bcrypt] python-dotenv python-multipart

```


3. **Environment Variables (.env):**
Create a `.env` file in the root directory:
```env
SECRET_KEY="your_super_secret_random_string"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=120

```


4. **Launch the Server:**
```bash
python main.py

```


The API will be available at `http://127.0.0.1:8000`.

---

## 📡 Key API Endpoints

### 🛍 Shop & AI

* `GET /boutique`: Returns a random selection of products with optional category filtering.
* `GET /recommend/{article_id}`: The AI engine. Returns the top 5 most similar items.
* `GET /categories`: List of all available product groups.

### 🔐 Authentication & User

* `POST /register`: Create a new account.
* `POST /login`: Returns a Bearer Token.
* `GET /cart`: View current user's saved items (requires Auth).

### 🛠 System

* `GET /static/{image_id}.jpg`: Serves product images directly.
* `GET /docs`: Interactive Swagger UI documentation.

