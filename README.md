# 🧠 Project Report: NLP Resume-to-Job Matcher

## 📌 Introduction
This project is a hands-on practice exercise for learning **NLP embeddings**, **vector similarity**, and **basic UI development** with Streamlit.  
The goal is to automatically match resumes with job descriptions based on their semantic similarity.

---

## 🎯 Objectives
- Use **OpenAI/MetisAI embeddings** to represent resumes and job postings as vectors.
- Calculate similarity between jobs and resumes using **Euclidean distance**.
- Build an interactive **Streamlit web app** to display the top 3 matching resumes for any selected job.
- Practice working with **JSON data parsing** and **local caching**.

---

## 🛠️ Methodology

### 1. Data Preparation
- `resumes.json` contains a list of resumes with summaries, experiences, and keywords.
- `job_opportunities.json` contains job descriptions and keywords.
- If keywords are missing, a basic keyword extractor is used as fallback.

### 2. Embedding
- Each keyword is converted into a numerical vector using the `text-embedding-3-small` model (or `ada-002` for legacy).
- Results are stored in `.embed_cache.json` to avoid re-computation.

### 3. Aggregation
- All keyword vectors for each resume and job are averaged into a **single mean vector** representing that item.

### 4. Similarity Calculation
- Euclidean distance is calculated between the job vector and each resume vector.
- Smaller distance = higher similarity.

### 5. Ranking
- Resumes are sorted by distance.
- The **top 3 matches** are displayed in the app.

---

## 💻 Implementation
- **Language:** Python 3  
- **Libraries:**  
  - `streamlit` – for UI  
  - `openai` – for embeddings  
  - `dotenv` – for environment variable management  
- **Key File:** `matcher_app.py`  
  - Loads data, extracts keywords, generates embeddings, ranks resumes, displays results.
- **UI:** Users select a job, click a button, and see ranked resumes with distance scores.

---

## 🧪 Results
- Successfully matches resumes with jobs based on semantic similarity.
- Demonstrates how embeddings can be used for resume screening, job recommendation, and information retrieval tasks.
- Provides a simple and visual way to understand vector similarity.

---

## 📈 Future Improvements
- Add **cosine similarity** as an alternative metric.
- Allow users to upload custom resumes and job descriptions.
- Deploy the app to **Streamlit Cloud** for easy sharing.
- Visualize similarity scores in a bar chart or radar plot.

---

## 🔑 Key Learnings
- How to work with **OpenAI embeddings** and aggregate them.
- How to cache results locally for performance.
- How to build a **Streamlit UI** to make NLP experiments interactive.
- The importance of `.gitignore` to protect private data like `.env` files.

---

## 📸 Demo
1. Select a job from the dropdown.
2. Click **🔍 Find Top 3 Matching Resumes**.
3. View the top matches and their similarity scores.
4. Tweak keywords in `resumes.json` and rerun to see updated results.

---

## 📚 Conclusion
This project successfully demonstrates an **end-to-end workflow** for NLP-based resume-job matching using embeddings.  
It provides a solid foundation for building more advanced systems like candidate recommendation engines or AI-powered HR tools.
