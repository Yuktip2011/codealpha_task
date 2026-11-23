# Smart FAQ Chatbot – AI-Powered Question Answering System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-purple)

**An intelligent, NLP-powered FAQ chatbot for college question answering.**

</div>

---

## 📌 Project Introduction

The **Smart FAQ Chatbot** is a fully functional AI-powered web application designed to answer college-related frequently asked questions in real time. It uses **Natural Language Processing (NLP)** techniques — specifically **TF-IDF Vectorization** and **Cosine Similarity** — to intelligently match a user's query to the most relevant entry in a predefined FAQ dataset.

Unlike simple keyword-based chatbots, this system understands the *meaning* behind questions even when phrased differently, making it a practical demonstration of applied Machine Learning in an academic context.

---

## 🎯 Project Objectives

- Provide instant, intelligent answers to college-related queries.
- Demonstrate real-world application of NLP preprocessing (tokenization, stopword removal, lemmatization).
- Implement TF-IDF vectorization and cosine similarity for semantic question matching.
- Build a modern, professional web UI with Streamlit.
- Serve as a complete, beginner-friendly reference project for college internships.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **NLP Preprocessing** | Lowercase, punctuation removal, tokenization, stopword filtering, lemmatization |
| 📐 **TF-IDF Vectorization** | Converts questions to numerical vectors (unigrams + bigrams) |
| 📏 **Cosine Similarity** | Finds the most semantically similar FAQ question |
| 🎯 **Confidence Scoring** | Displays High / Medium / Low confidence for each match |
| 💬 **Chat Interface** | Real-time chat bubbles with user/bot distinction |
| 💡 **Suggested Questions** | Clickable sample questions spread across all categories |
| 🔬 **NLP Visualizer** | Expandable step-by-step preprocessing breakdown |
| 🗑️ **Clear Chat** | Reset conversation history with one click |
| 📊 **Sidebar Dashboard** | Stats, categories, and "how it works" explanation |
| 📂 **Category Display** | Shows which FAQ category the matched answer belongs to |
| ⚠️ **Fallback Handling** | Graceful response when no suitable match is found |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Web frontend & UI framework |
| **NLTK** | NLP preprocessing (tokenization, stopwords, lemmatization) |
| **Scikit-learn** | TF-IDF vectorizer & cosine similarity |
| **Pandas** | FAQ dataset loading and management |
| **NumPy** | Numerical operations for similarity scoring |

---

## 🏗️ Project Architecture

```
Smart_FAQ_Chatbot/
│
├── app.py           ← Streamlit frontend: UI layout, chat rendering, event handling
├── chatbot.py       ← NLP engine: FAQ loading, TF-IDF model, cosine similarity
├── preprocess.py    ← Text preprocessing: clean, tokenize, remove stopwords, lemmatize
├── faqs.csv         ← FAQ dataset: 56+ Q&A pairs across 9 categories
├── requirements.txt ← Python dependency list
├── README.md        ← Project documentation (this file)
└── .gitignore       ← Git ignore rules
```

### Data Flow Diagram

```
User Input (text)
      │
      ▼
preprocess.py ── clean_text()
  • Lowercase
  • Remove special chars
  • Tokenize (NLTK)
  • Remove stopwords
  • Lemmatize
      │
      ▼
chatbot.py ── TF-IDF Vectorizer (scikit-learn)
  • Transform query to vector
  • Compute cosine similarity vs all FAQ vectors
  • Find best match above threshold
      │
      ▼
app.py ── Streamlit UI
  • Render answer in chat bubble
  • Show category + confidence score
  • Update chat history
```

---

## 📦 Installation Instructions

### Prerequisites
- Python 3.9 or higher installed
- pip package manager

### Step 1 – Clone or download the project

```bash
# If using Git
git clone <your-repo-url>
cd "Smart FAQ Chatbot"

# Or navigate to the project folder manually
cd "C:\Users\YourName\Desktop\Smart FAQ Chatbot"
```

### Step 2 – (Recommended) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – NLTK data is downloaded automatically

The first run of `preprocess.py` will silently download required NLTK packages:
- `punkt` – tokenizer models
- `stopwords` – English stopword list
- `wordnet` – lemmatizer lexical database

---

## ▶️ How to Run the Project

```bash
streamlit run app.py
```

The application will open automatically in your default browser at:
```
http://localhost:8501
```

---

## 📁 File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend. Contains all UI code: page config, custom CSS theme, chat history rendering, input handling, suggested questions, sidebar, and NLP step visualizer. |
| `chatbot.py` | Core NLP backend. Loads `faqs.csv`, preprocesses FAQ questions, builds the TF-IDF model, and exposes `get_response()` for question matching. |
| `preprocess.py` | Text cleaning utilities. Provides `clean_text()` (full pipeline) and `get_preprocessing_steps()` (for visualization). |
| `faqs.csv` | FAQ dataset with 56 entries across 9 categories. Columns: `category`, `question`, `answer`. Easy to extend — just add rows. |
| `requirements.txt` | Lists all Python packages required to run the project. |
| `README.md` | Complete project documentation. |
| `.gitignore` | Excludes `venv/`, `__pycache__/`, `.env`, and other non-essential files from version control. |

---

## 📸 Screenshots

> **To add screenshots:**
> 1. Run the application with `streamlit run app.py`
> 2. Take screenshots of the chat interface, sidebar, and preprocessing expander
> 3. Save them in a `screenshots/` folder and link here

| View | Description |
|---|---|
| ![Chat Interface](screenshots/chat.png) | Main chat interface with dark-blue/purple theme |
| ![Sidebar](screenshots/sidebar.png) | Sidebar showing stats and categories |
| ![NLP Steps](screenshots/nlp_steps.png) | Preprocessing step visualizer |

---

## 🗂️ FAQ Categories

The dataset covers **9 categories** with **56 FAQ entries**:

1. **College Admissions** – Eligibility, documents, process, intake
2. **Courses** – UG/PG programs, branches, duration, electives
3. **Fees** – Tuition, hostel, payment options, refund policy
4. **Examinations** – Structure, schedule, passing criteria, re-evaluation
5. **Library** – Hours, borrowing limits, e-resources, fines
6. **Placements** – Packages, companies, placement rate, training
7. **Scholarships** – Merit, EWS, education loans, fee waivers
8. **Campus Facilities** – Sports, canteen, medical, Wi-Fi, transport
9. **General Information** – Working hours, contact, affiliation, clubs, attendance

---

## 🔮 Future Improvements

- [ ] **Deep Learning Model** – Replace TF-IDF with sentence transformers (BERT/SBERT) for better semantic understanding
- [ ] **Dynamic FAQ Management** – Admin panel to add/edit/delete FAQs without touching the CSV
- [ ] **Multi-language Support** – Translate queries and answers for regional language users
- [ ] **Voice Input** – Allow users to speak their question using the Web Speech API
- [ ] **Analytics Dashboard** – Track most-asked questions and unanswered queries
- [ ] **Database Backend** – Replace CSV with SQLite or PostgreSQL for scalability
- [ ] **API Endpoint** – Expose the chatbot as a REST API for integration with other systems
- [ ] **Feedback System** – Let users rate answers (👍/👎) to improve matching over time

---

## 👨‍💻 Developer

| Field | Detail |
|---|---|
| **Project Type** | College Internship Project |
| **Domain** | Artificial Intelligence / NLP |
| **Complexity** | Intermediate |
| **Status** | ✅ Complete |

---

## 📄 License

This project is licensed under the **MIT License** – you are free to use, modify, and distribute it for educational and personal projects.

