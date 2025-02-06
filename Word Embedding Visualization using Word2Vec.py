import streamlit as st
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Title of the app
st.title("Word2Vec Visualization and Evaluation")

# Upload file section
uploaded_file = st.file_uploader("Upload a text file with sentences", type=["txt"])

if uploaded_file is not None:
    # Read the file and prepare sentences
    sentences = []
    for line in uploaded_file:
        sentences.append(line.decode("utf-8").strip().split())

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    st.success("✅ Word2Vec model trained successfully!")

    # Display vocabulary
    words = list(model.wv.index_to_key)
    st.write("📌 Vocabulary:", words)

    # Word similarity input
    word_input = st.text_input("Enter a word to find similar words:")
    if word_input:
        if word_input in words:
            similar_words = model.wv.most_similar(word_input, topn=5)
            st.write("🔍 Words similar to '{}':".format(word_input), similar_words)
        else:
            st.warning(f"⚠️ The word '{word_input}' is not in the vocabulary.")

    # PCA Visualization
    if st.button("Visualize with PCA"):
        word_vectors = model.wv[words]
        pca = PCA(n_components=2)
        word_vectors_2d = pca.fit_transform(word_vectors)

        plt.figure(figsize=(8, 6))
        plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

        for i, word in enumerate(words):
            plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

        plt.title("📌 Word Embeddings Visualization using PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid()
        st.pyplot(plt)

    # t-SNE Visualization
    if st.button("Visualize with t-SNE"):
        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        word_vectors_tsne = tsne.fit_transform(model.wv[words])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=word_vectors_tsne[:, 0], y=word_vectors_tsne[:, 1])

        for i, word in enumerate(words):
            plt.annotate(word, xy=(word_vectors_tsne[i, 0], word_vectors_tsne[i, 1]))

        plt.title("📌 Word Embeddings Visualization using t-SNE")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid()
        st.pyplot(plt)

    # Word analogy input        
    analogy_input = st.text_input("Enter analogy (e.g., king - man + woman):")
    if analogy_input:
        words_in_analogy = analogy_input.split()
        if len(words_in_analogy) == 3:
            try:
                analogy_result = model.wv.most_similar(positive=[words_in_analogy[0], words_in_analogy[2]], negative=[words_in_analogy[1]], topn=1)
                st.write("🔍 Analogy result for '{}':".format(analogy_input), analogy_result)
            except KeyError as e:
                st.error(f"⚠️ Error: {e}. Some words in the analogy are not in the vocabulary.")
        else:
            st.warning("⚠️ Please enter exactly three words for the analogy.")
