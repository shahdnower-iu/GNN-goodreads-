# 📚 Goodreads Book Recommendation System Using Graph Neural Networks (GNN)

## 📌 Project Overview

This project explores the application of **Graph Neural Networks (GNNs)** to build a book recommendation system using Goodreads data.

Unlike traditional recommendation systems that rely primarily on user-item rating matrices, this approach represents relationships between users and books as a graph, allowing the model to learn meaningful representations from connections and interactions.

The main objective is to leverage graph-based learning techniques to understand user preferences and recommend books that users may find interesting.

## 🎯 Project Objectives

- Explore and analyze Goodreads book and user interaction data.
- Understand the structure of user-book relationships.
- Represent the recommendation problem using graph structures.
- Apply Graph Neural Networks to learn user and book embeddings.
- Develop a recommendation approach based on graph representations.
- Evaluate the model's recommendation performance.

## 🧠 Methodology

The project follows these main stages:

### 1. Data Collection
- Load the Goodreads dataset.
- Explore available features and relationships.
- Identify relevant user, book, and rating information.

### 2. Data Preprocessing
- Handle missing values and duplicate records.
- Perform data cleaning and filtering.
- Prepare user-book interaction data.
- Transform the data into a graph-compatible format.

### 3. Graph Construction
Represent the dataset as a graph:

- **Nodes:** Users and books.
- **Edges:** Interactions between users and books.
- **Edge attributes:** Ratings or other interaction information, depending on the dataset.

### 4. Graph Neural Network Modeling

Apply Graph Neural Network techniques to learn representations of users and books by aggregating information from their neighboring nodes.

The learned embeddings can be used for recommendation tasks.

### 5. Evaluation

Evaluate the recommendation system using appropriate metrics based on the implemented recommendation task.

## 🛠️ Technologies & Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch / PyTorch Geometric (if used)
- NetworkX (if used)
- Jupyter Notebook / Google Colab

## 📂 Project Structure

```text
GNN-Goodreads/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── GNN_Goodreads.ipynb
│
├── src/
│   └── model.py
│
├── README.md
└── requirements.txt
```

*Adjust this structure to match the actual repository.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd GNN-Goodreads
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

Open the notebook:

```bash
jupyter notebook
```

Run the project notebook to explore the data, construct the graph, train the model, and evaluate its performance.

## 📊 Results

Results, visualizations, and model performance metrics will be documented here.

## 🔮 Future Improvements

- Experiment with different Graph Neural Network architectures.
- Improve recommendation quality through hyperparameter tuning.
- Explore advanced graph representation learning techniques.
- Incorporate additional book metadata.
- Develop an interactive book recommendation interface.

## 👩‍💻 Author

**Shahd**

Data Science | Machine Learning | Artificial Intelligence

---

⭐ If you find this project interesting, feel free to explore the repository!