"""
Command Line Interface for Chest X-ray Tokenizer
"""
import argparse
import json
from pathlib import Path
from tokenizer import ChestXRayTokenizer
from vector_store import VectorStore
from inference import PathologyInferenceEngine, create_sample_database


def main():
    parser = argparse.ArgumentParser(
        description="Chest X-ray Tokenizer - Create embeddings and infer pathologies"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode an image to vector embedding')
    encode_parser.add_argument('image', type=str, help='Path to chest X-ray image')
    encode_parser.add_argument('--output', type=str, help='Output file for embedding (JSON)')

    # Add reference command
    add_parser = subparsers.add_parser('add', help='Add a reference image to the database')
    add_parser.add_argument('image', type=str, help='Path to chest X-ray image')
    add_parser.add_argument('--pathologies', type=str, required=True,
                           help='Comma-separated pathologies (e.g., "Pneumonia,Effusion")')
    add_parser.add_argument('--patient-id', type=str, help='Patient identifier')
    add_parser.add_argument('--notes', type=str, help='Additional notes')

    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Infer pathologies from an image')
    infer_parser.add_argument('image', type=str, help='Path to chest X-ray image')
    infer_parser.add_argument('--k', type=int, default=5, help='Number of neighbors (default: 5)')
    infer_parser.add_argument('--show-neighbors', action='store_true',
                             help='Show detailed neighbor information')

    # Statistics command
    subparsers.add_parser('stats', help='Show database statistics')

    # Initialize sample database
    init_parser = subparsers.add_parser('init', help='Initialize sample database')
    init_parser.add_argument('--samples', type=int, default=20,
                            help='Number of sample embeddings (default: 20)')

    # Clear database
    subparsers.add_parser('clear', help='Clear all data from the database')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize engine
    print("Initializing Chest X-ray Tokenizer...")
    engine = PathologyInferenceEngine()

    # Execute commands
    if args.command == 'encode':
        print(f"Encoding image: {args.image}")
        embedding = engine.tokenizer.encode(args.image)
        print(f"✓ Generated embedding with dimension: {len(embedding)}")

        if args.output:
            output_data = {
                'image': args.image,
                'embedding_dimension': len(embedding),
                'embedding': embedding.tolist()
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f)
            print(f"✓ Saved to {args.output}")
        else:
            print(f"Embedding preview: {embedding[:5]}... (showing first 5 values)")

    elif args.command == 'add':
        pathologies = [p.strip() for p in args.pathologies.split(',')]
        print(f"Adding reference image: {args.image}")
        print(f"Pathologies: {pathologies}")

        embedding_id = engine.add_reference_image(
            image_path=args.image,
            pathologies=pathologies,
            patient_id=args.patient_id,
            notes=args.notes
        )

        engine.save_state()
        print(f"✓ Added with ID: {embedding_id}")

    elif args.command == 'infer':
        print(f"Inferring pathologies for: {args.image}")
        result = engine.infer_pathologies(
            image_path=args.image,
            return_neighbors=args.show_neighbors
        )

        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)

        if result['pathologies']:
            print(f"\nDetected Pathologies:")
            for pathology in result['pathologies']:
                confidence = result['confidence_scores'][pathology]
                print(f"  • {pathology}: {confidence:.2%}")
        else:
            print("\nNo pathologies detected or no similar images found")

        print(f"\nStatistics:")
        print(f"  • Neighbors found: {result['num_neighbors_found']}")
        print(f"  • Average similarity: {result['average_similarity']:.2%}")

        if args.show_neighbors and 'neighbors' in result:
            print(f"\n{'='*60}")
            print("SIMILAR IMAGES")
            print("="*60)
            for i, neighbor in enumerate(result['neighbors'], 1):
                print(f"\n{i}. {neighbor['image_path']}")
                print(f"   Similarity: {neighbor['similarity']:.2%}")
                print(f"   Pathologies: {', '.join(neighbor['pathologies'])}")
                print(f"   Patient: {neighbor['patient_id']}")

    elif args.command == 'stats':
        stats = engine.get_store_statistics()
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        print(f"\nTotal embeddings: {stats['total_embeddings']}")
        print(f"Unique patients: {stats['unique_patients']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")

        if stats['pathology_distribution']:
            print(f"\nPathology Distribution:")
            for pathology, count in sorted(
                stats['pathology_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  • {pathology}: {count}")
        else:
            print("\nNo pathologies in database")

    elif args.command == 'init':
        print(f"Initializing sample database with {args.samples} entries...")
        create_sample_database(engine, num_samples=args.samples)
        print("✓ Sample database created successfully")

    elif args.command == 'clear':
        confirm = input("Are you sure you want to clear the database? (yes/no): ")
        if confirm.lower() == 'yes':
            engine.vector_store.clear()
            engine.save_state()
            print("✓ Database cleared")
        else:
            print("Operation cancelled")


if __name__ == "__main__":
    main()
