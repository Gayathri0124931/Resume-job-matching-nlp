# AI-Based Resume Screening and Job Matching Using Natural Language Processing

##  Project Overview

The **AI-Based Resume Screening and Job Matching Using Natural Language Processing** project aims to automate the process of comparing resumes with job descriptions and identifying suitable job opportunities for candidates.

The project uses Natural Language Processing (NLP) techniques to preprocess resume and job-description text. **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert text into numerical representations, and **Cosine Similarity** is used to measure the similarity between a resume and available job descriptions.

The similarity scores are then used to **rank jobs for a selected resume**, with higher-scoring jobs appearing at the top.

---

##  Objectives

* Analyze resume and job-description datasets.
* Perform NLP-based text preprocessing.
* Clean and normalize resume and job-description text.
* Apply tokenization, stop-word removal, and lemmatization.
* Represent text using TF-IDF.
* Calculate resume-job similarity using cosine similarity.
* Rank available jobs based on similarity scores.
* Develop a baseline for future semantic job-matching systems.

---

##  Technologies Used

* **Python**
* **Pandas** – Data loading and analysis
* **NumPy** – Numerical operations
* **NLTK** – Natural Language Processing
* **Scikit-learn** – TF-IDF and cosine similarity
* **Jupyter Notebook** – Development and experimentation

---

## 📂 Dataset

The project uses a publicly available **Job-Resume Matching Dataset** from Hugging Face.

### Dataset Source

https://huggingface.co/datasets/nonameee12233/job-resume-matching

The dataset contains three CSV files:

### 1. `cv.csv`

Contains candidate/resume information such as:

* `candidate_id`
* `target_position`
* `resume_text`
* `clean_skills`
* `education_text`
* `experience_text`
* `years_experience`
* `inferred_category`
* `embedding`

### 2. `job.csv`

Contains job-related information such as:

* `job_id`
* `job_title`
* `category`
* `job_description`
* `job_text`
* `clean_skills`
* `education_text`
* `years_required`
* `embedding`

### 3. `matches.csv`

Contains candidate-job matching information including:

* `candidate_id`
* `job_id`
* `final_score`
* `semantic_score`
* `skill_score`
* `position_score`
* `category_score`
* `experience_score`
* `education_score`
* `candidate_rank`
* `candidate_rank_pct`

---

##  Project Workflow

The implemented Phase 1 system follows the workflow below:

```text
Resume and Job Dataset
        ↓
Data Exploration
        ↓
Text Preprocessing
        ↓
Feature Extraction
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Similarity Score
        ↓
Job Ranking
```

---

##  Data Exploration

The datasets were explored using Pandas.

The following operations were performed:

* Displayed sample records.
* Checked dataset dimensions.
* Inspected column names and data types.
* Checked missing values.
* Checked duplicate records.
* Examined candidate categories.
* Examined job categories.
* Examined candidate experience.
* Examined required job experience.

---

##  NLP Preprocessing

The textual data was processed using **NLTK**.

The preprocessing steps include:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove unnecessary special characters.
4. Remove extra whitespace.
5. Tokenize the text.
6. Remove stop words.
7. Apply lemmatization.

The main text fields considered include:

```text
resume_text
job_description
job_text
```

---

##  TF-IDF

TF-IDF was used as the baseline text representation technique.

TF-IDF converts textual documents into numerical vectors based on the importance of words within the documents.

The TF-IDF formula is:

```text
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

where:

```text
IDF(t) = log(N / DF(t))
```

This representation allows resumes and job descriptions to be compared mathematically.

---

##  Cosine Similarity

Cosine similarity is used to measure the similarity between a resume vector and a job-description vector.

The formula is:

```text
Similarity(A,B) = (A · B) / (||A|| ||B||)
```

The resulting score generally ranges from **0 to 1** for non-negative TF-IDF vectors.

* **Higher score** → greater textual similarity
* **Lower score** → lower textual similarity

For example:

```text
Resume-Job Similarity: 0.03765715349831636
```

The score represents the textual similarity between the selected resume and job description.

---

##  Job Ranking

After calculating cosine similarity between one selected resume and multiple jobs, the jobs are sorted in descending order of their similarity scores.

Example:

| Rank | Job   | Similarity |
| ---- | ----- | ---------- |
| 1    | Job A | 0.63       |
| 2    | Job B | 0.59       |
| 3    | Job C | 0.56       |
| 4    | Job D | 0.54       |
| 5    | Job E | 0.53       |

The job with the highest similarity score is ranked first.

---

##  Implementation

A simplified version of the similarity calculation is:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(tfidf_matrix)
```

For ranking jobs for a selected resume:

```python
scores = cosine_similarity(
    resume_vector,
    job_vectors
)[0]

job_results["similarity"] = scores

ranked_jobs = job_results.sort_values(
    by="similarity",
    ascending=False
)

print(ranked_jobs.head(10))
```

---

##  Concepts Covered

This project covers the following NLP and Machine Learning concepts:

* Natural Language Processing
* Text preprocessing
* Text normalization
* Tokenization
* Stop-word removal
* Lemmatization
* Feature extraction
* TF-IDF
* Vector representation
* Cosine similarity
* Text similarity
* Resume screening
* Job matching
* Similarity-based ranking
* Exploratory Data Analysis using Pandas

---

##  Suggested Project Structure

```text
AI-Resume-Job-Matching/
│
├── data/
│   ├── cv.csv
│   ├── job.csv
│   └── matches.csv
│
├── notebooks/
│   └── resume_job_matching.ipynb
│
├── README.md
│
└── requirements.txt
```

---

##  Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-LINK>
```

Navigate to the project folder:

```bash
cd AI-Resume-Job-Matching
```

Install the required Python libraries:

```bash
pip install pandas numpy nltk scikit-learn jupyter
```

If required, download the NLTK resources:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

##  How to Run

1. Open the project in **Jupyter Notebook**.
2. Load `cv.csv`, `job.csv`, and `matches.csv`.
3. Perform data exploration.
4. Apply NLP preprocessing to the text fields.
5. Generate TF-IDF vectors.
6. Calculate cosine similarity.
7. Select a resume.
8. Compare the resume with available jobs.
9. Sort jobs by similarity score.
10. Display the top-ranked jobs.

---

##  Current Phase

### Phase 1 – Baseline NLP Job Matching

The current implementation establishes a baseline using:

```text
NLP Preprocessing
        +
TF-IDF
        +
Cosine Similarity
        +
Job Ranking
```

This provides an interpretable starting point for automated resume-job matching.

---

##  Future Improvements

Future versions of the project can include:

* Semantic embeddings for better contextual understanding.
* Skill-based matching.
* Education-based matching.
* Experience-based matching.
* Job-category matching.
* Position/title matching.
* Multi-factor matching scores.
* Transformer-based models.
* Interactive resume upload interface.
* Personalized job recommendations.
* Improved ranking and evaluation methods.

---

##  Limitations

The current system uses TF-IDF as the primary text representation method. Therefore, it mainly captures **word-level/lexical similarity** and may not fully understand semantic relationships between different terms.

For example, related terms may have different representations even when they have similar meanings.

The similarity score should therefore be considered an **initial matching indicator**, not a final recruitment decision.

---

##  Team Members

* **Amina Munna** – 25MCAR0178
* **Isha K S** – 25MCAR0010
* **Asher Jacob Dani** – 25MCAR0005
* **Vansh Parashar** – 25MCAR0220

---

##  References

1. Li, et al. *Competence-Level Prediction and Resume & Job Description Matching Using Context-Aware Transformer Models*. Proceedings of EMNLP, 2020.
   https://aclanthology.org/2020.emnlp-main.679/

2. Rojas-Galeano, S., et al. *A Bibliometric Perspective on AI Research for Job-Résumé Matching*. 2022.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9550515/

3. Sinha, S., Akhtar, M. S., and Kumar, A. *Resume Screening Using Natural Language Processing and Machine Learning: A Systematic Review*. 2021.

4. Modak, et al. *A Review of Resume Analysis and Job Description Matching Using Machine Learning*. 2024.

5. *ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Negative Mining*. Findings of ACL 2025.
   https://aclanthology.org/2025.findings-acl.661/

---

##  Project Status

**Phase 1: Completed**

The baseline resume-job matching pipeline has been implemented using NLP preprocessing, TF-IDF, cosine similarity, and similarity-based job ranking.
