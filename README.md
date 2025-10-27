# Chest X-ray Tokenizer System

A deep learning-based system for encoding chest X-ray images into vector embeddings and inferring pathologies using nearest neighbor search.

## Overview

This system allows you to:
- **Tokenize** chest X-ray images into fixed-size vector embeddings
- **Store** embeddings in an efficient vector database with FAISS indexing
- **Search** for similar chest X-rays using nearest neighbor search
- **Infer** pathologies based on similar reference images

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Chest X-ray    │─────▶│  Deep Learning   │─────▶│     Vector      │
│     Image       │      │  Tokenizer       │      │   Embedding     │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                              (DenseNet/            (1024-dim vector)
                              ResNet/ViT)
                                                            │
                                                            ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Pathology     │◀─────│  k-NN Search     │◀─────│  FAISS Vector   │
│   Inference     │      │  + Scoring       │      │     Store       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

## Features

- **Multiple Model Architectures**: Support for DenseNet121, ResNet50, and Vision Transformers
- **Efficient Search**: FAISS-based nearest neighbor search for fast retrieval
- **Flexible Storage**: HDF5 and CSV-based metadata management
- **REST API**: FastAPI-based API for easy integration
- **CLI Interface**: Command-line tool for batch processing
- **Pathology Classes**: Support for 15 common chest X-ray pathologies

## Supported Pathologies

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia
- No Finding

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### For CPU-only Installation

If you don't have a GPU, modify `requirements.txt`:
```bash
# Replace torch and torchvision with CPU versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### 1. Initialize Sample Database

Create a sample database for testing:

```bash
python cli.py init --samples 50
```

### 2. Add Reference Image

Add a chest X-ray with known pathologies:

```bash
python cli.py add path/to/xray.png \
    --pathologies "Pneumonia,Effusion" \
    --patient-id "P001" \
    --notes "Follow-up study"
```

### 3. Infer Pathologies

Analyze a new chest X-ray:

```bash
python cli.py infer path/to/query.png \
    --k 5 \
    --show-neighbors
```

### 4. View Statistics

Check database statistics:

```bash
python cli.py stats
```

## Usage

### Python API

```python
from tokenizer import ChestXRayTokenizer
from vector_store import VectorStore
from inference import PathologyInferenceEngine

# Initialize
engine = PathologyInferenceEngine()

# Add reference image
engine.add_reference_image(
    image_path="xray_001.png",
    pathologies=["Pneumonia"],
    patient_id="P001"
)

# Infer pathologies
result = engine.infer_pathologies("query_xray.png")
print(result['pathologies'])
print(result['confidence_scores'])
```

### REST API

Start the API server:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Infer Pathologies**
```bash
curl -X POST "http://localhost:8000/infer" \
  -F "file=@chest_xray.png" \
  -F "return_neighbors=true"
```

**Add Reference Image**
```bash
curl -X POST "http://localhost:8000/add_reference" \
  -F "file=@chest_xray.png" \
  -F "pathologies=Pneumonia,Effusion" \
  -F "patient_id=P001"
```

**Get Statistics**
```bash
curl "http://localhost:8000/statistics"
```

**Encode Image**
```bash
curl -X POST "http://localhost:8000/encode" \
  -F "file=@chest_xray.png"
```

### CLI Usage

```bash
# Encode an image to embedding
python cli.py encode xray.png --output embedding.json

# Add reference image
python cli.py add xray.png --pathologies "Pneumonia"

# Infer pathologies
python cli.py infer query.png --k 5 --show-neighbors

# View statistics
python cli.py stats

# Initialize sample database
python cli.py init --samples 100

# Clear database
python cli.py clear
```

## Configuration

Edit `config.py` to customize:

```python
# Model Configuration
MODEL_NAME = "densenet121"  # Options: densenet121, resnet50, vit_base_patch16_224
EMBEDDING_DIM = 1024
IMAGE_SIZE = (224, 224)

# Nearest Neighbor Configuration
K_NEIGHBORS = 5
SIMILARITY_THRESHOLD = 0.7

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
```

## Project Structure

```
chest-xray-tokenizer/
├── config.py              # Configuration settings
├── tokenizer.py           # Image encoding module
├── vector_store.py        # Vector storage and search
├── inference.py           # Pathology inference engine
├── api.py                 # FastAPI REST API
├── cli.py                 # Command-line interface
├── example_usage.py       # Example scripts
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Reference images (created on first run)
├── embeddings/            # Vector store and metadata (created on first run)
├── models/                # Model checkpoints (created on first run)
└── .cache/                # Temporary files (created on first run)
```

## How It Works

### 1. Image Encoding

The tokenizer uses a pre-trained deep learning model (DenseNet121 by default) to convert chest X-ray images into 1024-dimensional vector embeddings:

```
Chest X-ray (224x224) → DenseNet121 → L2 Normalization → Vector (1024-dim)
```

### 2. Vector Storage

Embeddings are stored in a FAISS index for efficient similarity search. Metadata (pathologies, patient IDs, etc.) is stored in a CSV file:

```
FAISS Index: Fast similarity search (L2 distance)
Metadata: CSV with image paths, pathologies, timestamps
```

### 3. Pathology Inference

When querying a new image:
1. Encode the image to a vector
2. Find k nearest neighbors in the database
3. Calculate weighted pathology scores based on neighbor similarities
4. Return ranked pathologies with confidence scores

```python
confidence(pathology) = Σ(similarity_i * has_pathology_i) / Σ(similarity_i)
```

## Advanced Usage

### Custom Model

Use a different model architecture:

```python
from tokenizer import ChestXRayTokenizer

# Use ResNet50
tokenizer = ChestXRayTokenizer(
    model_name="resnet50",
    embedding_dim=2048
)

# Use Vision Transformer
tokenizer = ChestXRayTokenizer(
    model_name="vit_base_patch16_224",
    embedding_dim=768
)
```

### Fine-tuning

Fine-tune on your own chest X-ray dataset:

```python
# 1. Load base tokenizer
tokenizer = ChestXRayTokenizer(pretrained=True)

# 2. Fine-tune on your dataset
# (implement custom training loop with your labeled data)

# 3. Save fine-tuned model
tokenizer.save_model("models/finetuned_model.pth")

# 4. Load for inference
tokenizer.load_model("models/finetuned_model.pth")
```

### Batch Processing

Process multiple images efficiently:

```python
image_paths = ["xray1.png", "xray2.png", "xray3.png"]

# Encode batch
embeddings = tokenizer.encode_batch(image_paths)

# Add to vector store
vector_store.add_embeddings(
    embeddings=embeddings,
    image_paths=image_paths,
    pathologies=[["Pneumonia"], ["No Finding"], ["Effusion"]]
)
```

## Performance Considerations

### GPU Acceleration

The system automatically uses GPU if available:

```python
# Check device
print(tokenizer.device)  # 'cuda' or 'cpu'

# Force CPU
tokenizer = ChestXRayTokenizer(device='cpu')
```

### Embedding Dimension vs. Accuracy

Higher dimensions may capture more information but increase storage:

- **512-dim**: Fast, compact, good for large databases
- **1024-dim**: Balanced (recommended)
- **2048-dim**: Best quality, more storage

### Search Speed

FAISS IndexFlatL2 provides exact search. For larger databases (>1M images), consider:

```python
# Use approximate search for speed
import faiss
index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
```

## Datasets

To train or evaluate on standard chest X-ray datasets:

- **ChestX-ray14**: 112,120 frontal-view X-rays (14 pathologies)
- **CheXpert**: 224,316 chest radiographs (14 observations)
- **MIMIC-CXR**: 377,110 chest X-rays with free-text reports
- **PadChest**: 160,000 images from 67,000 patients

## Medical Disclaimer

⚠️ **IMPORTANT**: This is a research/educational tool and should NOT be used for clinical diagnosis without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical decisions.

## Limitations

- Trained on general ImageNet features (better performance possible with medical imaging pre-training)
- Inference based on similarity, not causal diagnosis
- Requires labeled reference database for accurate results
- Performance depends on quality and diversity of reference images

## Future Improvements

- [ ] Add DICOM file format support
- [ ] Implement attention visualization for interpretability
- [ ] Support for multi-view chest X-rays
- [ ] Integration with radiology reports (NLP)
- [ ] Active learning for continuous improvement
- [ ] Model ensemble for improved accuracy
- [ ] Support for segmentation masks
- [ ] Privacy-preserving federated learning

## Contributing

Contributions are welcome! Areas for improvement:
- Medical imaging pre-trained models
- Additional pathology classes
- Better evaluation metrics
- Web interface
- Docker containerization

## License

This project is for educational and research purposes. Consult with legal experts before using in production medical settings.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{chest_xray_tokenizer,
  title={Chest X-ray Tokenizer: Deep Learning-based Image Encoding and Pathology Inference},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/chest-xray-tokenizer}
}
```

## References

- Wang et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database"
- Irvin et al. (2019). "CheXpert: A Large Chest Radiograph Dataset"
- Johnson et al. (2019). "MIMIC-CXR: A large publicly available database of labeled chest radiographs"

## Support

For questions or issues:
- Open an issue on GitHub
- Check the example scripts in `example_usage.py`
- Review API documentation at `/docs` when running the server

---

**Built with**: PyTorch, FAISS, FastAPI, and ❤️ for medical AI
