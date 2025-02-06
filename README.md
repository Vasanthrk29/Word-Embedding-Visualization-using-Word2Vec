# Word-Embedding-Visualization-using-Word2Vec
SLP Assignment 1 - Word2Vec Model Training and Visualization
Overview
This project focuses on training a Word2Vec model using the gensim library. It includes word embedding visualization using PCA and t-SNE and evaluates the model through word similarity and analogy tasks.

Features
✅ Train Word2Vec Model on a sample dataset.
🔍 Word Embedding Exploration - Retrieve vectors and find similar words.
📊 Visualization:
PCA for dimensionality reduction.
t-SNE for advanced 2D projection.
📈 Evaluation Methods:
Intrinsic evaluation (word similarity, analogies).
Placeholder for extrinsic evaluation (downstream tasks like classification).
💾 Model Saving & Loading: Save and reload the trained model.
Installation
To run the script, install the required dependencies:

pip install --upgrade gensim nltk matplotlib seaborn scikit-learn
How to Run
Clone the repository:
git clone https://github.com/your-username/your-repository.git
cd your-repository
Install dependencies:
pip install -r requirements.txt
Run the script:
python 201_slp_assignment1.py
View model training outputs and visualizations.
Expected Outputs
Console logs for vocabulary, word similarities, and analogy results.
Saved model file: word2vec_model.bin.
PCA and t-SNE visualization plots.
Repository Structure
├── 201_slp_assignment1.py   # Main script
├── word2vec_model.bin       # Saved Word2Vec model
├── README.md                # Project documentation
└── requirements.txt         # Required dependencies
Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

License
This project is licensed under the MIT License.

Author
Vasanth RK
