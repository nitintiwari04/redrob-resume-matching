import math

# -----------------------------
# SKILL ALIASES
# -----------------------------

SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",

    # Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",

    # Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# -----------------------------
# RESUME DATA
# -----------------------------

resumes = {
    "Arjun Sharma": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",

    "Priya Nair": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",

    "Rahul Gupta": "Java, Spring Boot, MySql, Microservices, Docker, kubernates",

    "Sneha Patel": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",

    "Vikram Singh": "C++, Algoritms, Data Structure, competitive programming, python",

    "Ananya Krishnan": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",

    "Karan Mehta": "Python, Sklearn, XGboost, feature engineering, SQL, tableau",

    "Deepika Rao": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",

    "Aditya Kumar": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",

    "Meera Iyer": "python, R, statistics, ML, regression, clustering, Power-BI"
}

# -----------------------------
# JOB DESCRIPTIONS
# -----------------------------

job_descriptions = {
    "JD-1": """
    Python, Machine Learning, Deep Learning,
    TensorFlow, PyTorch, SQL, Data Visualization,
    NLP, BERT, Feature Engineering, Statistics
    """,

    "JD-2": """
    Java, Spring Boot, MySQL, PostgreSQL,
    Microservices, Docker, Kubernetes,
    REST API, CI/CD, Redis
    """,

    "JD-3": """
    JavaScript, React, Vue, TypeScript,
    REST API, HTML/CSS, Node.js,
    GraphQL, Redux, Jest, AWS
    """
}

# -----------------------------
# NORMALIZATION
# -----------------------------

def normalize_skills(skill_string):
    tokens = skill_string.lower().split(",")

    normalized = []

    for token in tokens:
        token = token.strip()

        if token in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[token])

    return list(set(normalized))


normalized_resumes = {}

for name, skills in resumes.items():
    normalized_resumes[name] = normalize_skills(skills)

# -----------------------------
# VOCABULARY
# -----------------------------

vocabulary = set()

for skills in normalized_resumes.values():
    vocabulary.update(skills)

vocabulary = sorted(vocabulary)

# -----------------------------
# DOCUMENT FREQUENCY
# -----------------------------

df = {}

for skill in vocabulary:
    count = 0

    for skills in normalized_resumes.values():
        if skill in skills:
            count += 1

    df[skill] = count

# -----------------------------
# IDF
# -----------------------------

idf = {}

TOTAL_RESUMES = 10

for skill in vocabulary:
    idf[skill] = math.log(TOTAL_RESUMES / df[skill])

# -----------------------------
# TF-IDF
# -----------------------------

resume_vectors = {}

for name, skills in normalized_resumes.items():

    vector = {}

    total_skills = len(skills)

    for vocab_skill in vocabulary:

        if vocab_skill in skills:
            tf = 1 / total_skills
            vector[vocab_skill] = tf * idf[vocab_skill]

        else:
            vector[vocab_skill] = 0

    resume_vectors[name] = vector

# -----------------------------
# JD BINARY VECTORS
# -----------------------------

jd_vectors = {}

for jd_name, jd_skills in job_descriptions.items():

    normalized_jd = normalize_skills(jd_skills)

    vector = {}

    for vocab_skill in vocabulary:

        if vocab_skill in normalized_jd:
            vector[vocab_skill] = 1
        else:
            vector[vocab_skill] = 0

    jd_vectors[jd_name] = vector

# -----------------------------
# COSINE SIMILARITY
# -----------------------------

def cosine_similarity(vec1, vec2):

    dot_product = 0

    for skill in vocabulary:
        dot_product += vec1[skill] * vec2[skill]

    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)

# -----------------------------
# RANKING
# -----------------------------

results = {}

for jd_name, jd_vector in jd_vectors.items():

    scores = []

    for candidate, resume_vector in resume_vectors.items():

        similarity = cosine_similarity(resume_vector, jd_vector)

        scores.append((candidate, similarity))

    scores.sort(key=lambda x: (-x[1], x[0]))

    results[jd_name] = scores[:3]

# -----------------------------
# OUTPUT
# -----------------------------

for jd_name, top_candidates in results.items():

    print(f"\n{jd_name} Results:")

    output = []

    for candidate, score in top_candidates:
        output.append(f"{candidate}({round(score, 2)})")

    print(", ".join(output))