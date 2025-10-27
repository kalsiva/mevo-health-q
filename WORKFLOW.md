# Chest X-ray Tokenizer - Workflow Diagram

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CHEST X-RAY TOKENIZER SYSTEM                      │
└──────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │   Input: Chest X-ray    │
                    │    (PNG/JPG/DICOM)      │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Image Preprocessing   │
                    │  • Resize to 224x224    │
                    │  • Grayscale → RGB      │
                    │  • Normalize            │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Deep Learning Model    │
                    │   (DenseNet121/         │
                    │    ResNet50/ViT)        │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Vector Embedding       │
                    │  (1024-dim vector)      │
                    │  [0.23, -0.45, ...]     │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌────────────────────┐        ┌──────────────────────┐
    │  Store in Database │        │  Search for Similar  │
    │   (FAISS Index)    │        │      Images          │
    │                    │        │   (k-NN Search)      │
    │ • Add embedding    │        └──────────┬───────────┘
    │ • Add metadata     │                   │
    │ • Save to disk     │                   ▼
    └────────────────────┘        ┌──────────────────────┐
                                  │  Pathology Inference │
                                  │  • Weight by         │
                                  │    similarity        │
                                  │  • Calculate scores  │
                                  │  • Rank pathologies  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │      Results         │
                                  │ • Pathologies: [...]  │
                                  │ • Confidence: X%     │
                                  │ • Similar cases      │
                                  └──────────────────────┘
```

## Workflow 1: Adding Reference Images

```
Step 1: Prepare Data
┌──────────────────────────┐
│  Collect Chest X-rays    │
│  with known diagnoses    │
└────────────┬─────────────┘
             │
             ▼
Step 2: Add to Database
┌──────────────────────────┐
│  python cli.py add       │
│  image.png               │
│  --pathologies           │
│  "Pneumonia"             │
└────────────┬─────────────┘
             │
             ▼
Step 3: System Processing
┌──────────────────────────┐
│  • Encode image          │
│  • Generate embedding    │
│  • Store in FAISS        │
│  • Save metadata         │
└────────────┬─────────────┘
             │
             ▼
Step 4: Ready for Search
┌──────────────────────────┐
│  Database now contains   │
│  this reference image    │
└──────────────────────────┘
```

## Workflow 2: Inferring Pathologies

```
Step 1: Query Image
┌──────────────────────────┐
│  Upload new chest X-ray  │
│  (unknown pathology)     │
└────────────┬─────────────┘
             │
             ▼
Step 2: Encode
┌──────────────────────────┐
│  python cli.py infer     │
│  query.png               │
│  --k 5                   │
└────────────┬─────────────┘
             │
             ▼
Step 3: Search Database
┌──────────────────────────┐
│  Find k=5 most similar   │
│  images in database      │
└────────────┬─────────────┘
             │
             ▼
Step 4: Calculate Scores
┌──────────────────────────┐
│  Similar Image 1 (90%)   │
│  → Pneumonia             │
│  Similar Image 2 (85%)   │
│  → Pneumonia, Effusion   │
│  Similar Image 3 (80%)   │
│  → Pneumonia             │
│  ...                     │
└────────────┬─────────────┘
             │
             ▼
Step 5: Weighted Inference
┌──────────────────────────┐
│  Pneumonia: 85%          │
│  (appears in 3/5 images) │
│                          │
│  Effusion: 35%           │
│  (appears in 1/5 images) │
└────────────┬─────────────┘
             │
             ▼
Step 6: Return Results
┌──────────────────────────┐
│  Most likely:            │
│  • Pneumonia (85%)       │
│  • Effusion (35%)        │
│                          │
│  + Details about         │
│    similar cases         │
└──────────────────────────┘
```

## Three Ways to Use the System

### 1. Command Line Interface (CLI)

```
┌─────────────────────────────────────────────┐
│  Terminal                                    │
│                                              │
│  $ python cli.py add image.png \            │
│      --pathologies "Pneumonia"              │
│  ✓ Added with ID: 42                        │
│                                              │
│  $ python cli.py infer query.png            │
│  Detected: Pneumonia (87%)                  │
└─────────────────────────────────────────────┘

Best for:
• Batch processing
• Testing
• Automation scripts
```

### 2. Python API (Programmatic)

```
┌─────────────────────────────────────────────┐
│  Python Script                               │
│                                              │
│  from inference import                      │
│    PathologyInferenceEngine                 │
│                                              │
│  engine = PathologyInferenceEngine()        │
│                                              │
│  result = engine.infer_pathologies(         │
│      "query.png"                            │
│  )                                          │
│                                              │
│  print(result['pathologies'])               │
└─────────────────────────────────────────────┘

Best for:
• Integration with other Python code
• Custom pipelines
• Data science workflows
```

### 3. REST API (Web Service)

```
┌─────────────────────────────────────────────┐
│  HTTP Client (curl/browser/app)             │
│                                              │
│  POST /infer                                │
│  Content-Type: multipart/form-data          │
│  file: [binary image data]                  │
│                                              │
│  ↓                                          │
│                                              │
│  Response:                                  │
│  {                                          │
│    "pathologies": ["Pneumonia"],            │
│    "confidence_scores": {                   │
│      "Pneumonia": 0.87                      │
│    }                                        │
│  }                                          │
└─────────────────────────────────────────────┘

Best for:
• Web applications
• Mobile apps
• Microservices architecture
```

## Data Flow Example

### Example: Diagnosing a New Patient

```
1. Hospital receives patient with respiratory symptoms
   └─→ Take chest X-ray

2. Doctor uploads X-ray to system
   └─→ POST /infer with image file

3. System encodes the X-ray
   └─→ DenseNet121 produces 1024-dim vector

4. FAISS searches 10,000 reference X-rays
   └─→ Finds 5 most similar cases in 10ms

5. Similar cases:
   • Case A (95% similar): Pneumonia, prescribed antibiotics, recovered
   • Case B (92% similar): Pneumonia + Effusion, hospitalized 5 days
   • Case C (89% similar): Pneumonia, elderly patient, ICU care
   • Case D (87% similar): Pneumonia, responded well to treatment
   • Case E (85% similar): Viral Pneumonia, supportive care

6. System calculates weighted scores:
   • Pneumonia: 94% confidence (appears in 5/5 cases)
   • Effusion: 18% confidence (appears in 1/5 cases)

7. Doctor receives:
   • Predicted pathology: Pneumonia (94%)
   • 5 similar historical cases with outcomes
   • Can review similar X-rays for visual comparison

8. Doctor makes informed decision:
   • Order confirmatory tests
   • Consider treatment options
   • Reference similar case outcomes
```

## Performance Characteristics

```
┌─────────────────────────┬──────────────────────┐
│ Operation               │ Typical Time         │
├─────────────────────────┼──────────────────────┤
│ Encode single image     │ 50-200ms (GPU)       │
│ (CPU)                   │ 200-500ms (CPU)      │
├─────────────────────────┼──────────────────────┤
│ Search database         │ <10ms (1K images)    │
│ (FAISS k-NN)           │ <50ms (100K images)  │
├─────────────────────────┼──────────────────────┤
│ Add to database         │ <5ms per image       │
├─────────────────────────┼──────────────────────┤
│ Full inference pipeline │ 100-300ms            │
└─────────────────────────┴──────────────────────┘

Storage Requirements:
• Each embedding: 4KB (1024 floats × 4 bytes)
• 1,000 images: ~4 MB
• 100,000 images: ~400 MB
• 1,000,000 images: ~4 GB
```

## System Components Interaction

```
┌──────────────┐
│   cli.py     │──┐
└──────────────┘  │
                  │
┌──────────────┐  │    ┌──────────────────┐
│   api.py     │──┼───→│  inference.py    │
└──────────────┘  │    └────────┬─────────┘
                  │             │
┌──────────────┐  │             │
│ custom.py    │──┘             │
└──────────────┘                │
                    ┌───────────┴───────────┐
                    │                       │
         ┌──────────▼─────────┐  ┌─────────▼──────────┐
         │   tokenizer.py     │  │  vector_store.py   │
         │                    │  │                    │
         │ • Load model       │  │ • FAISS index      │
         │ • Preprocess       │  │ • Metadata DB      │
         │ • Generate vector  │  │ • Search           │
         └────────────────────┘  └────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     config.py         │
                    │                       │
                    │ • Model settings      │
                    │ • Paths               │
                    │ • Parameters          │
                    └───────────────────────┘
```

## Quick Reference

**Initialize & Test:**
```bash
python quickstart_test.py     # Run complete test
python cli.py init            # Create sample DB
python cli.py stats           # Check status
```

**Build Database:**
```bash
python cli.py add image.png --pathologies "Pneumonia"
```

**Make Predictions:**
```bash
python cli.py infer query.png --show-neighbors
```

**Start API:**
```bash
python api.py                 # http://localhost:8000
```

**Python Integration:**
```python
from inference import PathologyInferenceEngine
engine = PathologyInferenceEngine()
result = engine.infer_pathologies("image.png")
```

---

For detailed step-by-step instructions, see: **QUICKSTART.md**
