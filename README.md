# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Sara Gomaa Elmahrouk                 |
| Project Name     | CTF Pattern Vault                    |
| GitHub Username  | saraelmahrouk                        |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

CTF Pattern Vault is an AI-powered study companion for CTF (Capture the Flag) practice. It ingests CTF writeups — both public ones scraped from GitHub and the user's own solved challenges — and turns each one into a structured "pattern card" capturing the vulnerability class, tools used, difficulty, and key insight. These pattern cards are embedded and stored in a vector database, letting the app act as a personal, ever-growing knowledge base of CTF techniques.

Rather than just handing over answers, the app is built around helping the user actually get better at solving challenges — offering graduated hints, topic-level coaching summaries, and direct help, all grounded in real past writeups instead of general model knowledge alone.

---

# ✨ Features

1. Direct Help — describe what you're stuck on, and get a grounded answer pulled from similar 2. past writeups.

2. Graduated Hints — instead of a direct answer, get three progressively revealing hints (category nudge → tool/technique nudge → near-complete walkthrough), so the app supports practice rather than just handing over solutions.

3. Topic Coaching — ask about a broad CTF topic (e.g. "XOR encryption challenges") and get a synthesized summary of everything the corpus knows about it, powered by unsupervised clustering of pattern cards into technique families.

4. Custom Writeup Ingestion — upload or paste your own writeups directly through the app; they're automatically extracted into pattern cards and added to the corpus and clustering model, so the system keeps learning from your own solves over time.

5. Cluster-Biased Retrieval — retrieval is aware of technique "families" discovered via clustering, boosting cards from the most relevant cluster while still falling back gracefully to plain similarity search.

6. Incremental Clustering — cluster assignments update as new writeups are added, using MiniBatchKMeans' partial-fit support, with periodic re-evaluation of cluster count (k) as the corpus grows.

7. Offline and Online mode - where it uses Cloud APIs to access a stronger model with better retrieval with the availability of internet, and the abilitiy to toggle that to an offline version that uses a locally installed model that is still works for when internet is not an available option and if the machine you're using can withstand downloading a local offline model.

---

# 🛠️ Technologies Used

. Python

. LangChain (langchain_core, langchain_classic, langchain_huggingface, langchain_community) — ..chains, structured output parsing, document loaders

. Groq Cloud — hosted LLM inference (currently a 70B-class model) for extraction, RAG, hinting, and coaching

. Local Qwen/Qwen2.5-7B-Instruct (4-bit quantized via bitsandbytes) — original local inference path; planned for an offline mode via QLoRA fine-tuning

. sentence-transformers/all-MiniLM-L6-v2 (via HuggingFaceEmbeddings) — fully local embeddings, no API calls

. FAISS — vector store for pattern card retrieval

. scikit-learn (MiniBatchKMeans, PCA) — incremental clustering of pattern cards into technique families

. Streamlit — web UI, with a custom retro pixel/CRT visual theme
ngrok — planned, for exposing the Streamlit app for live sharing/demos

---

# ⚙️ Installation

```bash
git clone <this-repo>
cd CTF-Pattern-Vault
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Set up a `.env` file with your API keys/tokens (e.g. `HF_TOKEN` for gated model downloads, Groq API key), matching what `config.py` expects.


---

# 🚀 Usage

Launch the app with:

```bash
streamlit run app.py
```

From the sidebar, navigate between:
- **Direct Help** — ask a specific question about what you're stuck on
- **Hints** — get progressively revealed hints for a challenge
- **Coaching** — ask about a general CTF topic for a knowledge summary
- **Upload Writeup** — paste or upload your own solved writeup to grow the corpus

To add new public writeups to the corpus later, just use the tab in the website called "Upload Writeups" and it will directly add your writeup to the corpus and aid in the learning process!

---

# 📸 Demo
<img width="1920" height="1080" alt="Screenshot (153)" src="https://github.com/user-attachments/assets/170cf9fb-c6be-461a-9685-928bc6820848" />
<img width="1920" height="1080" alt="Screenshot (152)" src="https://github.com/user-attachments/assets/7d1ea8d9-f171-4ef9-ac0c-7979ccb91b8f" />
<img width="1920" height="1080" alt="Screenshot (147)" src="https://github.com/user-attachments/assets/c6ec4dc5-7510-466a-82e8-63d01be60804" />
<img width="1920" height="1080" alt="Screenshot (149)" src="https://github.com/user-attachments/assets/97a8fb7a-3713-476c-b3ee-f80ce23f5c4b" />
<img width="1920" height="806" alt="Screenshot (145)" src="https://github.com/user-attachments/assets/a54c60bb-3c8f-48e7-8646-c137b8c9ab9d" />
<img width="1920" height="850" alt="Screenshot (146)" src="https://github.com/user-attachments/assets/936b8136-7739-408f-870c-8fbe3528189e" />
<img width="1920" height="1080" alt="Screenshot (148)" src="https://github.com/user-attachments/assets/72f62411-4cba-4ee3-8609-bccd2bbda892" />
<img width="1920" height="1080" alt="Screenshot (150)" src="https://github.com/user-attachments/assets/85eed689-4635-4061-b2a9-94cdd88b7d07" />
<img width="1920" height="1080" alt="Screenshot (151)" src="https://github.com/user-attachments/assets/35237580-efc9-4b19-98e7-017a43c5643e" />
<img width="1920" height="817" alt="Screenshot (144)" src="https://github.com/user-attachments/assets/e5eb018c-b702-4b91-a2c9-120668f6c647" />


---

# 📈 Results

* Successfully ingested and processed a corpus of public CTF writeups spanning multiple competitions and categories (web, crypto, forensics, misc, reversing).

* Built four distinct, working LLM-backed features (direct help, graduated hints, topic coaching, custom writeup ingestion) sharing one underlying pattern-card corpus.

* Implemented incremental, persistent clustering (MiniBatchKMeans + PCA) that updates as the corpus grows, rather than requiring a full retrain on every addition.

* Migrated from fully local inference (Mistral-7B, 4-bit quantized) to Groq Cloud for generation, meaningfully improving output quality and reducing fabricated/hallucinated technical details.

---

# 🔮 Future Improvements

* A bigger corpus for stronger retrieval of information and help 

* Adding memory to the model so that it remembers past questions and hiccups you've had before, therefore better helping you with future questions

* A better check for the writeups being ingested into the website so that it doesn't corrupt the corpus of the website

* Automatic reclustering algorithm that routinely checks for changes in the corpus that could affect the optimal k and redesigns the clusters rather than assigning topics to already existing clusters.

* Add login and authentication to the app and make user use their own groq API not mine

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
