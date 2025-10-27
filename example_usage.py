"""
Example Usage of Chest X-ray Tokenizer System

This script demonstrates how to use the tokenizer system programmatically
"""
from tokenizer import ChestXRayTokenizer
from vector_store import VectorStore
from inference import PathologyInferenceEngine, create_sample_database
import numpy as np


def example_1_basic_encoding():
    """
    Example 1: Basic image encoding
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Image Encoding")
    print("="*60)

    # Initialize tokenizer
    tokenizer = ChestXRayTokenizer()

    # For this example, we'll create a dummy image
    # In practice, you would use: embedding = tokenizer.encode("path/to/xray.png")
    print("\nNote: In production, use: tokenizer.encode('path/to/xray.png')")
    print("For this example, we're creating a random embedding for demonstration")

    # Create dummy embedding
    dummy_embedding = np.random.randn(tokenizer.embedding_dim)
    print(f"\n✓ Generated embedding with dimension: {len(dummy_embedding)}")
    print(f"✓ Embedding preview: {dummy_embedding[:5]}...")


def example_2_building_database():
    """
    Example 2: Building a reference database
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Building a Reference Database")
    print("="*60)

    # Initialize components
    tokenizer = ChestXRayTokenizer()
    vector_store = VectorStore()

    # Simulate adding multiple reference images
    print("\nAdding reference images to database...")

    reference_data = [
        ("xray_001.png", ["Pneumonia"]),
        ("xray_002.png", ["Atelectasis", "Effusion"]),
        ("xray_003.png", ["Cardiomegaly"]),
        ("xray_004.png", ["No Finding"]),
        ("xray_005.png", ["Pneumothorax"]),
    ]

    # Generate dummy embeddings (in practice, use tokenizer.encode_batch())
    embeddings = np.random.randn(len(reference_data), tokenizer.embedding_dim).astype('float32')

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    image_paths = [item[0] for item in reference_data]
    pathologies = [item[1] for item in reference_data]
    patient_ids = [f"P{i:05d}" for i in range(len(reference_data))]

    # Add to vector store
    ids = vector_store.add_embeddings(
        embeddings=embeddings,
        image_paths=image_paths,
        pathologies=pathologies,
        patient_ids=patient_ids
    )

    print(f"✓ Added {len(ids)} reference images")
    print(f"✓ Assigned IDs: {ids}")

    # Get statistics
    stats = vector_store.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  • Total embeddings: {stats['total_embeddings']}")
    print(f"  • Pathology distribution: {stats['pathology_distribution']}")

    return vector_store


def example_3_inference():
    """
    Example 3: Pathology inference using nearest neighbors
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Pathology Inference")
    print("="*60)

    # Initialize inference engine
    engine = PathologyInferenceEngine()

    # Create sample database
    print("\nCreating sample database...")
    create_sample_database(engine, num_samples=20)

    # Simulate a query image
    print("\nPerforming inference on a query image...")
    print("Note: In production, use: engine.infer_pathologies('path/to/query.png')")

    # Create a dummy query embedding (simulate a chest X-ray with pneumonia)
    query_embedding = np.random.randn(engine.tokenizer.embedding_dim)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search for neighbors
    distances, indices, metadata = engine.vector_store.search(
        query_embedding=query_embedding,
        k=5
    )

    print(f"\n✓ Found {len(metadata)} similar images")
    print("\nNearest Neighbors:")

    for i, (dist, idx, row) in enumerate(zip(distances, indices, metadata.iterrows()), 1):
        _, data = row
        print(f"\n{i}. {data['image_path']}")
        print(f"   Similarity: {data['similarity']:.2%}")
        print(f"   Pathologies: {data['pathologies']}")
        print(f"   Patient: {data['patient_id']}")


def example_4_batch_processing():
    """
    Example 4: Batch processing multiple images
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)

    # Initialize tokenizer
    tokenizer = ChestXRayTokenizer()

    # Simulate batch of images
    print("\nProcessing batch of images...")
    print("Note: In production, use: tokenizer.encode_batch(image_paths)")

    num_images = 10
    print(f"Processing {num_images} images in batch...")

    # Create dummy embeddings (in practice, use tokenizer.encode_batch())
    batch_embeddings = np.random.randn(num_images, tokenizer.embedding_dim).astype('float32')

    # Normalize
    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    batch_embeddings = batch_embeddings / norms

    print(f"✓ Generated {batch_embeddings.shape[0]} embeddings")
    print(f"✓ Embedding shape: {batch_embeddings.shape}")

    # Calculate similarity matrix (as an example analysis)
    similarity_matrix = np.dot(batch_embeddings, batch_embeddings.T)
    avg_similarity = (similarity_matrix.sum() - num_images) / (num_images * (num_images - 1))

    print(f"\nBatch Statistics:")
    print(f"  • Average inter-image similarity: {avg_similarity:.2%}")


def example_5_custom_pipeline():
    """
    Example 5: Custom processing pipeline
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Processing Pipeline")
    print("="*60)

    # Create custom pipeline
    print("\nBuilding custom pipeline...")

    # Initialize components
    tokenizer = ChestXRayTokenizer(model_name="densenet121")
    vector_store = VectorStore(embedding_dim=1024)
    engine = PathologyInferenceEngine(
        tokenizer=tokenizer,
        vector_store=vector_store,
        k_neighbors=3,
        similarity_threshold=0.6
    )

    print(f"✓ Tokenizer: {tokenizer.model_name}")
    print(f"✓ Embedding dimension: {tokenizer.embedding_dim}")
    print(f"✓ Device: {tokenizer.device}")
    print(f"✓ k-neighbors: {engine.k_neighbors}")
    print(f"✓ Similarity threshold: {engine.similarity_threshold}")

    print("\n✓ Custom pipeline ready for processing")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("CHEST X-RAY TOKENIZER - EXAMPLE USAGE")
    print("="*80)

    try:
        example_1_basic_encoding()
        example_2_building_database()
        example_3_inference()
        example_4_batch_processing()
        example_5_custom_pipeline()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext steps:")
        print("1. Replace dummy embeddings with real images using tokenizer.encode()")
        print("2. Build your reference database with labeled chest X-rays")
        print("3. Use the API (api.py) or CLI (cli.py) for production use")
        print("4. Fine-tune the model on chest X-ray datasets for better performance")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
