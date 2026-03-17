
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

base_folder = "my_knowledge"
supported_extensions = (".txt", ".pdf", ".docx")

# -------- ENSURE FOLDER STRUCTURE EXISTS --------
for subfolder in ["pdf", "txt", "docx"]:
    os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)

docs = []

# -------- LOAD DOCUMENTS --------
for root, dirs, files in os.walk(base_folder):
    for file in files:
        path = os.path.join(root, file)

        try:
            if file.endswith(".txt"):
                loader = TextLoader(path)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = path
            docs.extend(loaded_docs)
            print(f"  📄 Loaded: {path}")

        except Exception as e:
            print(f"  ⚠️  Skipped {path}: {e}")

# -------- CHECK DOCS FOUND --------
if not docs:
    print(f"\n⚠️  No documents found in '{base_folder}'.")
    print(f"   Add files to the appropriate subfolder and re-run:")
    print(f"   {base_folder}/pdf/    ← PDF files")
    print(f"   {base_folder}/txt/    ← TXT files")
    print(f"   {base_folder}/docx/   ← DOCX files")
    print(f"\n   Supported formats: {', '.join(supported_extensions)}")
    exit(0)  # Clean exit, not a crash

print(f"\n✅ Loaded {len(docs)} documents")

# -------- CHUNKING --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunked_docs = splitter.split_documents(docs)

if not chunked_docs:
    print("⚠️  Documents were found but produced no chunks — files may be empty.")
    exit(0)  # Clean exit, not a crash

print(f"✅ Created {len(chunked_docs)} chunks")

# -------- EMBEDDINGS & DB --------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

db = Chroma.from_documents(
    chunked_docs,
    embeddings,
    persist_directory="./db",
)

print("✅ Database built successfully!")