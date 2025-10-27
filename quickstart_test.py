"""
Quickstart Test Script
Run this to test the Chest X-ray Tokenizer system without real images
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config
from inference import PathologyInferenceEngine, create_sample_database


def create_dummy_xray_image(filename, width=512, height=512, text="Chest X-ray"):
    """
    Create a dummy grayscale image that simulates a chest X-ray

    Args:
        filename: Output filename
        width: Image width
        height: Image height
        text: Text to overlay on image
    """
    # Create grayscale image with random noise (simulating X-ray texture)
    img_array = np.random.randint(100, 200, (height, width), dtype=np.uint8)

    # Add some structure (simulating lungs/ribs)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    # Left lung
    left_lung_mask = ((x - center_x + 80)**2 + (y - center_y)**2) < 15000
    img_array[left_lung_mask] = np.clip(img_array[left_lung_mask] - 30, 0, 255)

    # Right lung
    right_lung_mask = ((x - center_x - 80)**2 + (y - center_y)**2) < 15000
    img_array[right_lung_mask] = np.clip(img_array[right_lung_mask] - 30, 0, 255)

    # Create PIL image
    img = Image.fromarray(img_array, mode='L')

    # Add text overlay
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a default font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Calculate text position (bottom center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2
    text_y = height - 40

    draw.text((text_x, text_y), text, fill=255, font=font)

    # Save
    img.save(filename)
    print(f"✓ Created dummy X-ray: {filename}")


def main():
    """
    Run a complete test of the system
    """
    print("="*70)
    print("CHEST X-RAY TOKENIZER - QUICKSTART TEST")
    print("="*70)

    # Step 1: Create sample images directory
    print("\n[Step 1] Creating sample images directory...")
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    print(f"✓ Directory created: {sample_dir}/")

    # Step 2: Generate dummy chest X-ray images
    print("\n[Step 2] Generating dummy chest X-ray images...")
    test_images = [
        (f"{sample_dir}/chest_xray_001.png", "Patient A - Pneumonia"),
        (f"{sample_dir}/chest_xray_002.png", "Patient B - Effusion"),
        (f"{sample_dir}/chest_xray_003.png", "Patient C - Normal"),
        (f"{sample_dir}/query_image.png", "Query Image"),
    ]

    for filename, label in test_images:
        create_dummy_xray_image(filename, text=label)

    # Step 3: Initialize the inference engine
    print("\n[Step 3] Initializing Chest X-ray Tokenizer...")
    engine = PathologyInferenceEngine()
    print(f"✓ Model: {engine.tokenizer.model_name}")
    print(f"✓ Device: {engine.tokenizer.device}")
    print(f"✓ Embedding dimension: {engine.tokenizer.embedding_dim}")

    # Step 4: Create initial reference database
    print("\n[Step 4] Creating reference database with sample embeddings...")
    create_sample_database(engine, num_samples=15)

    # Step 5: Add real reference images
    print("\n[Step 5] Adding dummy X-ray images to database...")
    reference_images = [
        (f"{sample_dir}/chest_xray_001.png", ["Pneumonia"], "P001"),
        (f"{sample_dir}/chest_xray_002.png", ["Effusion", "Atelectasis"], "P002"),
        (f"{sample_dir}/chest_xray_003.png", ["No Finding"], "P003"),
    ]

    for img_path, pathologies, patient_id in reference_images:
        embedding_id = engine.add_reference_image(
            image_path=img_path,
            pathologies=pathologies,
            patient_id=patient_id
        )
        print(f"  ✓ Added {img_path} (ID: {embedding_id})")

    # Save state
    engine.save_state()

    # Step 6: Show database statistics
    print("\n[Step 6] Database Statistics:")
    print("-" * 70)
    stats = engine.get_store_statistics()
    print(f"  • Total embeddings: {stats['total_embeddings']}")
    print(f"  • Unique patients: {stats['unique_patients']}")
    print(f"  • Embedding dimension: {stats['embedding_dimension']}")

    if stats['pathology_distribution']:
        print(f"\n  Pathology Distribution:")
        for pathology, count in sorted(
            stats['pathology_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:  # Show top 5
            print(f"    • {pathology}: {count}")

    # Step 7: Perform inference on query image
    print("\n[Step 7] Testing pathology inference on query image...")
    print("-" * 70)
    query_path = f"{sample_dir}/query_image.png"

    result = engine.infer_pathologies(
        image_path=query_path,
        return_neighbors=True
    )

    print(f"\nQuery Image: {query_path}")
    print(f"\nInference Results:")

    if result['pathologies']:
        print(f"  Detected Pathologies:")
        for pathology in result['pathologies']:
            confidence = result['confidence_scores'][pathology]
            print(f"    • {pathology}: {confidence:.2%}")
    else:
        print(f"  • No pathologies detected (or no similar images found)")

    print(f"\n  Statistics:")
    print(f"    • Neighbors found: {result['num_neighbors_found']}")
    print(f"    • Average similarity: {result['average_similarity']:.2%}")

    if result.get('neighbors'):
        print(f"\n  Top 3 Similar Images:")
        for i, neighbor in enumerate(result['neighbors'][:3], 1):
            print(f"    {i}. {neighbor['image_path']}")
            print(f"       Similarity: {neighbor['similarity']:.2%}")
            print(f"       Pathologies: {', '.join(neighbor['pathologies'])}")

    # Step 8: Test encoding
    print("\n[Step 8] Testing direct image encoding...")
    print("-" * 70)
    embedding = engine.tokenizer.encode(query_path)
    print(f"✓ Generated embedding vector")
    print(f"  • Shape: {embedding.shape}")
    print(f"  • Preview: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
    print(f"  • L2 Norm: {np.linalg.norm(embedding):.4f} (should be ~1.0 for normalized)")

    # Final summary
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY! ✓")
    print("="*70)

    print("\n📋 What was tested:")
    print("  ✓ Created dummy chest X-ray images")
    print("  ✓ Initialized tokenizer model")
    print("  ✓ Built reference database")
    print("  ✓ Added labeled images")
    print("  ✓ Performed pathology inference")
    print("  ✓ Retrieved similar images")
    print("  ✓ Encoded images to vectors")

    print("\n📝 Next Steps:")
    print("  1. View sample images in: sample_images/")
    print("  2. Try CLI commands:")
    print("     python cli.py stats")
    print("     python cli.py infer sample_images/query_image.png --show-neighbors")
    print("  3. Start API server:")
    print("     python api.py")
    print("  4. Replace dummy images with real chest X-rays!")

    print("\n💡 To use with real images:")
    print("  python cli.py add /path/to/real_xray.png --pathologies 'Pneumonia'")
    print("  python cli.py infer /path/to/query_xray.png")

    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure you installed all dependencies:")
        print("   pip install -r requirements.txt")
