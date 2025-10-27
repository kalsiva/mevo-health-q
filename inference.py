"""
Pathology Inference Engine for Chest X-rays
Uses nearest neighbor search to infer pathologies from similar images
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
import config
from tokenizer import ChestXRayTokenizer
from vector_store import VectorStore


class PathologyInferenceEngine:
    """
    Infers pathologies from chest X-ray images using k-nearest neighbors
    """

    def __init__(
        self,
        tokenizer: ChestXRayTokenizer = None,
        vector_store: VectorStore = None,
        k_neighbors: int = config.K_NEIGHBORS,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD
    ):
        """
        Initialize the inference engine

        Args:
            tokenizer: ChestXRayTokenizer instance
            vector_store: VectorStore instance
            k_neighbors: Number of nearest neighbors to consider
            similarity_threshold: Minimum similarity for a neighbor to be considered
        """
        self.tokenizer = tokenizer or ChestXRayTokenizer()
        self.vector_store = vector_store or VectorStore()
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold

    def infer_pathologies(
        self,
        image_path: str,
        return_neighbors: bool = True
    ) -> Dict:
        """
        Infer pathologies from a chest X-ray image

        Args:
            image_path: Path to the chest X-ray image
            return_neighbors: Whether to include neighbor details in the result

        Returns:
            Dictionary containing inferred pathologies and confidence scores
        """
        # Generate embedding for the query image
        print(f"Encoding image: {image_path}")
        query_embedding = self.tokenizer.encode(image_path)

        # Search for nearest neighbors
        print(f"Searching for {self.k_neighbors} nearest neighbors...")
        distances, indices, neighbor_metadata = self.vector_store.search(
            query_embedding,
            k=self.k_neighbors
        )

        # Filter by similarity threshold
        neighbor_metadata = neighbor_metadata[
            neighbor_metadata['similarity'] >= self.similarity_threshold
        ]

        if len(neighbor_metadata) == 0:
            return {
                'pathologies': [],
                'confidence_scores': {},
                'message': 'No similar images found above similarity threshold',
                'neighbors': []
            }

        # Extract pathologies from neighbors
        pathology_votes = []
        for pathology_str in neighbor_metadata['pathologies']:
            if pathology_str and pathology_str != "No Finding":
                pathology_votes.extend(pathology_str.split(','))

        # Calculate weighted pathology scores
        pathology_scores = self._calculate_pathology_scores(
            neighbor_metadata['pathologies'].tolist(),
            neighbor_metadata['similarity'].tolist()
        )

        # Sort pathologies by confidence
        sorted_pathologies = sorted(
            pathology_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = {
            'pathologies': [p[0] for p in sorted_pathologies if p[1] > 0.2],
            'confidence_scores': dict(sorted_pathologies),
            'num_neighbors_found': len(neighbor_metadata),
            'average_similarity': float(neighbor_metadata['similarity'].mean())
        }

        if return_neighbors:
            result['neighbors'] = self._format_neighbors(neighbor_metadata)

        return result

    def _calculate_pathology_scores(
        self,
        pathologies_list: List[str],
        similarities: List[float]
    ) -> Dict[str, float]:
        """
        Calculate weighted pathology scores based on neighbor similarities

        Args:
            pathologies_list: List of pathology strings from neighbors
            similarities: List of similarity scores for each neighbor

        Returns:
            Dictionary mapping pathology names to confidence scores
        """
        pathology_scores = {}

        for pathology_str, similarity in zip(pathologies_list, similarities):
            if pathology_str and pathology_str != "No Finding":
                pathologies = pathology_str.split(',')
                for pathology in pathologies:
                    pathology = pathology.strip()
                    if pathology not in pathology_scores:
                        pathology_scores[pathology] = 0
                    pathology_scores[pathology] += similarity

        # Normalize scores
        total_similarity = sum(similarities)
        if total_similarity > 0:
            pathology_scores = {
                k: v / total_similarity
                for k, v in pathology_scores.items()
            }

        return pathology_scores

    def _format_neighbors(self, neighbor_metadata: pd.DataFrame) -> List[Dict]:
        """
        Format neighbor information for output

        Args:
            neighbor_metadata: DataFrame with neighbor metadata

        Returns:
            List of neighbor dictionaries
        """
        neighbors = []
        for _, row in neighbor_metadata.iterrows():
            neighbors.append({
                'id': int(row['id']),
                'image_path': row['image_path'],
                'pathologies': row['pathologies'].split(',') if row['pathologies'] else [],
                'similarity': float(row['similarity']),
                'distance': float(row['distance']),
                'patient_id': row['patient_id']
            })
        return neighbors

    def batch_infer(
        self,
        image_paths: List[str],
        return_neighbors: bool = False
    ) -> List[Dict]:
        """
        Infer pathologies for multiple images

        Args:
            image_paths: List of image paths
            return_neighbors: Whether to include neighbor details

        Returns:
            List of inference results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.infer_pathologies(image_path, return_neighbors)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'pathologies': []
                })
        return results

    def add_reference_image(
        self,
        image_path: str,
        pathologies: List[str],
        patient_id: str = None,
        notes: str = None
    ) -> int:
        """
        Add a reference image to the vector store

        Args:
            image_path: Path to the chest X-ray image
            pathologies: List of confirmed pathologies
            patient_id: Optional patient identifier
            notes: Optional notes

        Returns:
            ID of the added embedding
        """
        # Generate embedding
        print(f"Encoding reference image: {image_path}")
        embedding = self.tokenizer.encode(image_path)

        # Add to vector store
        ids = self.vector_store.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            image_paths=[image_path],
            pathologies=[pathologies],
            patient_ids=[patient_id] if patient_id else None,
            notes=[notes] if notes else None
        )

        return ids[0]

    def get_store_statistics(self) -> Dict:
        """
        Get statistics about the reference database

        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_statistics()

    def save_state(self):
        """
        Save the vector store and model state
        """
        print("Saving vector store...")
        self.vector_store.save()
        print("Inference engine state saved")

    def load_state(self):
        """
        Load the vector store state
        """
        print("Loading vector store...")
        self.vector_store.load()
        print("Inference engine state loaded")


def create_sample_database(engine: PathologyInferenceEngine, num_samples: int = 10):
    """
    Create a sample database with synthetic embeddings for testing

    Args:
        engine: PathologyInferenceEngine instance
        num_samples: Number of sample embeddings to create
    """
    print(f"Creating sample database with {num_samples} entries...")

    # Sample pathologies from config
    sample_pathologies = config.PATHOLOGY_CLASSES[:10]

    # Generate random embeddings (in production, these would be real images)
    embeddings = np.random.randn(num_samples, config.EMBEDDING_DIM).astype('float32')

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Create metadata
    image_paths = [f"sample_data/xray_{i:03d}.png" for i in range(num_samples)]
    pathologies_list = [
        [np.random.choice(sample_pathologies)]
        for _ in range(num_samples)
    ]
    patient_ids = [f"P{i:05d}" for i in range(num_samples)]

    # Add to vector store
    engine.vector_store.add_embeddings(
        embeddings=embeddings,
        image_paths=image_paths,
        pathologies=pathologies_list,
        patient_ids=patient_ids
    )

    engine.save_state()
    print(f"Sample database created with {num_samples} entries")


if __name__ == "__main__":
    # Example usage
    print("Testing PathologyInferenceEngine...")

    # Initialize engine
    engine = PathologyInferenceEngine()

    # Create a sample database for testing
    create_sample_database(engine, num_samples=20)

    # Get statistics
    stats = engine.get_store_statistics()
    print("\nDatabase Statistics:")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Unique patients: {stats['unique_patients']}")
    print(f"Pathology distribution: {stats['pathology_distribution']}")
