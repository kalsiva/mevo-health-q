"""
Configuration file for Chest X-ray Tokenizer System
"""
import os

# Model Configuration
MODEL_NAME = "densenet121"  # Options: densenet121, resnet50, vit_base_patch16_224
EMBEDDING_DIM = 1024  # Dimension of the output embeddings
IMAGE_SIZE = (224, 224)  # Input image size
PRETRAINED = True  # Use ImageNet pre-trained weights

# Data Configuration
DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"
CACHE_DIR = ".cache"

# Create directories if they don't exist
for directory in [DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Pathology Classes (based on common chest X-ray datasets like ChestX-ray14)
PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

# Nearest Neighbor Configuration
K_NEIGHBORS = 5  # Number of nearest neighbors to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for pathology inference

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# Vector Store Configuration
VECTOR_STORE_PATH = os.path.join(EMBEDDINGS_DIR, "vector_store.h5")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.idx")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.csv")
