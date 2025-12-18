<img width="702" height="277" alt="Screenshot 2025-12-18 105543" src="https://github.com/user-attachments/assets/23cd897f-3936-49e6-a175-a91063f125af" />


# RAG From Scratch : Guide Pratique de la Génération Augmentée par Récupération

Ce dépôt contient un notebook complet détaillant l'implémentation progressive d'un système de **Génération Augmentée par Récupération (RAG)**. L'objectif est de passer d'une recherche textuelle basique à un pipeline de production utilisant des bases de données vectorielles et des LLM locaux.

---

## 🏗️ Architecture du Projet

Le projet est divisé en quatre étapes clés pour comprendre chaque composant d'un système RAG :

### 1. Recherche Sémantique & Diversité (TF-IDF + MMR)
* **Objectif** : Comprendre la récupération d'information sans LLM.
* **Concepts** : Similarité cosinus et **Maximal Marginal Relevance (MMR)** pour équilibrer la pertinence et la diversité des résultats récupérés.

### 2. Pipeline RAG "From Scratch"
* **Composants** :
    * **Chunking** : Découpage intelligent avec `RecursiveCharacterTextSplitter`.
    * **Embeddings** : Utilisation de `SentenceTransformers` (`all-MiniLM-L6-v2`).
    * **Vector Store** : Indexation locale rapide avec **FAISS** (`IndexFlatIP`).
    * **LLM Local** : Inférence sur CPU avec **Llama.cpp** (Modèle Qwen 2.5).



### 3. RAG avec LangChain
* **Objectif** : Simplifier le code en utilisant un framework industriel.
* **Outils** : Utilisation de `RetrievalQA` pour "câbler" automatiquement le processeur de texte, le store FAISS et le modèle de langue.

### 4. RAG de Production (Weaviate Cloud)
* **Objectif** : Déporter le stockage vers une base de données vectorielle externe.
* **Technique** : Intégration avec **Weaviate Cloud Services (WCS)** pour gérer l'indexation HNSW et les métadonnées à grande échelle.

---

## 🚀 Cas d'Application : Analyse de PDF
Le notebook inclut une implémentation finale permettant d'interroger des fichiers PDF complexes (exemple utilisé : *Cybersécurité - Fondements et pratiques avancées*).
* Extraction de texte avec `pypdf`.
* Interface de chat en ligne de commande pour poser des questions sur le document.

---

## 🛠️ Installation

### Prérequis
* Python 3.10+
* Google Colab (recommandé pour l'accélération TPU/GPU)

### Dépendances
```bash
pip install numpy weaviate-client sentence-transformers llama-cpp-python pypdf faiss-cpu langchain
