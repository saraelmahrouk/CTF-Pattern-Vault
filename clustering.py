import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from vectorstore import load as load_vectorstore
from sklearn.metrics import silhouette_score
from vectorstore import load
from sklearn.decomposition import PCA
import joblib


CLUSTER_MODEL_PATH = "ctf_corpus/cluster_model.pkl"
PCA_MODEL_PATH = "ctf_corpus/pca_model.pkl"

print("Loading vectorstore...")
vectorstore = load()
num_vectors = vectorstore.index.ntotal
embeddings = np.array([vectorstore.index.reconstruct(i) for i in range(num_vectors)])


def predict_query_cluster(query_embedding):
    kmeans, pca = load_models()
    if kmeans is None:
        raise ValueError("No cluster model found — run fit_initial() first.")

    query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    reduced_query = pca.transform(query_array)
    cluster_id = kmeans.predict(reduced_query)[0]
    return cluster_id

def get_doc_cluster_ids(vectorstore, kmeans, pca):
    """Returns {source_path: cluster_id} for every card currently in the corpus."""
    num_vectors = vectorstore.index.ntotal
    vector_matrix = np.array([vectorstore.index.reconstruct(i) for i in range(num_vectors)])
    reduced = pca.transform(vector_matrix)
    labels = kmeans.predict(reduced)

    doc_cluster_map = {}
    for i in range(num_vectors):
        doc_id = vectorstore.index_to_docstore_id[i]
        doc = vectorstore.docstore.search(doc_id)
        source = doc.metadata.get("source")
        doc_cluster_map[source] = labels[i]
    return doc_cluster_map

def cluster_biased_retrieve(vectorstore, query, k=4, pool_size=10):
    kmeans, pca = load_models()

    query_embedding = vectorstore.embeddings.embed_query(query)
    predicted_cluster = predict_query_cluster(query_embedding)

    candidates = vectorstore.similarity_search_with_score(query, k=pool_size)
    doc_cluster_map = get_doc_cluster_ids(vectorstore, kmeans, pca)

    def sort_key(candidate):
        doc, score = candidate
        same_cluster = doc_cluster_map.get(doc.metadata.get("source")) == predicted_cluster
        return (0 if same_cluster else 1, score)

    reranked = sorted(candidates, key=sort_key)
    return [doc for doc, score in reranked[:k]]

def get_cluster_topic_docs(vectorstore, query, max_cards=5):
    kmeans, pca = load_models()
    query_embedding = vectorstore.embeddings.embed_query(query)
    predicted_cluster = predict_query_cluster(query_embedding)

    doc_cluster_map = get_doc_cluster_ids(vectorstore, kmeans, pca)

    matching_docs = []
    for doc_id, docstore_doc in vectorstore.docstore._dict.items():
        source = docstore_doc.metadata.get("source")
        if doc_cluster_map.get(source) == predicted_cluster:
            matching_docs.append(docstore_doc)

    return matching_docs[:max_cards]

def find_optimal_k(max_k=15):
    # 1. Load the existing LangChain FAISS vectorstore
    print("Loading vectorstore...")
    vectorstore = load() 
    
    # 2. Extract raw embeddings array from the underlying FAISS index
    num_vectors = vectorstore.index.ntotal
    print(f"Found {num_vectors} pattern cards in the index.")
    
    # FAISS indices allow reconstructing the original vectors
    embeddings = np.array([vectorstore.index.reconstruct(i) for i in range(num_vectors)])

    pca = PCA(n_components=5)
    embeddings_reduced = pca.fit_transform(embeddings)

    # 3. Calculate Inertia and Silhouette for k=2 to max_k
    k_values = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_values:
        # MiniBatchKMeans is fast and memory efficient
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings_reduced)
        
        inertias.append(kmeans.inertia_)
        score = silhouette_score(embeddings_reduced, labels)
        silhouette_scores.append(score)
        
    # 4. Plot the results side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Elbow (Inertia)
    ax1.plot(k_values, inertias, 'bo-', linewidth=2)
    ax1.set_title("Elbow Method (Inertia)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Lower is tighter)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Silhouette Score
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2)
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Score (Closer to 1 is better)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def cluster_and_display():
    """Loads the SAVED cluster model and shows current groupings — does not train anything."""
    kmeans, pca = load_models()
    if kmeans is None:
        raise ValueError("No cluster model found — run fit_initial() first.")

    print("Loading vectorstore...")
    vectorstore = load()
    num_vectors = vectorstore.index.ntotal

    vector_matrix = np.array([vectorstore.index.reconstruct(i) for i in range(num_vectors)])

    docs = []
    for i in range(num_vectors):
        doc_id = vectorstore.index_to_docstore_id[i]
        doc = vectorstore.docstore.search(doc_id)
        docs.append(doc)

    # reuse the SAVED PCA projection — never re-fit here
    reduced_vectors = pca.transform(vector_matrix)

    # use predict, not fit_predict — we're reading the existing model, not training it
    labels = kmeans.predict(reduced_vectors)

    clusters = {i: [] for i in range(kmeans.n_clusters)}
    for idx, label in enumerate(labels):
        doc = docs[idx]
        source_path = doc.metadata.get("source", f"Writeup_{idx}")
        filename = source_path.replace("\\", "/").split("/")[-1]
        snippet = doc.page_content[:120].replace("\n", " ").strip()
        clusters[label].append((filename, snippet))

    print("==========================================")
    print("        TECHNIQUE FAMILY CLUSTERS         ")
    print("==========================================")
    for cluster_id, items in clusters.items():
        print(f"\n📂 Cluster {cluster_id} ({len(items)} cards):")
        print("-" * 40)
        for name, snippet in items:
            print(f"  • [{name}]")
            print(f"    └─ {snippet}...")

def save_models(kmeans, pca):
    joblib.dump(kmeans, CLUSTER_MODEL_PATH)
    joblib.dump(pca, PCA_MODEL_PATH)

def load_models():
    if not os.path.exists(CLUSTER_MODEL_PATH):
        return None, None
    kmeans = joblib.load(CLUSTER_MODEL_PATH)
    pca = joblib.load(PCA_MODEL_PATH)
    return kmeans, pca

def fit_initial(embeddings, k=7, pca_components=5):
    pca = PCA(n_components=pca_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(reduced)

    save_models(kmeans, pca)
    return kmeans, pca

def update_with_new(new_embeddings):
    kmeans, pca = load_models()
    if kmeans is None:
        raise ValueError("No existing cluster model found — run fit_initial() first.")

    # Project new embeddings into the SAME PCA space as before — do NOT re-fit PCA
    reduced_new = pca.transform(new_embeddings)

    # partial_fit updates the existing centroids incrementally,
    # instead of retraining from scratch on everything
    kmeans.partial_fit(reduced_new)

    save_models(kmeans, pca)
    return kmeans, pca

if __name__ == "__main__":
    fit_initial(embeddings)
    # update_with_new(embeddings)
    # find_optimal_k()
    # cluster_and_display()



