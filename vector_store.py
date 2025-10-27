"""
Vector Storage and Nearest Neighbor Search for Chest X-ray Embeddings
"""
import numpy as np
import pandas as pd
import faiss
import h5py
from typing import List, Dict, Tuple, Optional
import config
import os
from datetime import datetime


class VectorStore:
    """
    Manages storage and retrieval of chest X-ray embeddings with FAISS indexing
    """

    def __init__(
        self,
        embedding_dim: int = config.EMBEDDING_DIM,
        index_path: str = config.FAISS_INDEX_PATH,
        metadata_path: str = config.METADATA_PATH
    ):
        """
        Initialize the vector store

        Args:
            embedding_dim: Dimension of the embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Initialize FAISS index (using L2 distance, can be changed to inner product)
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Metadata storage
        self.metadata = pd.DataFrame(columns=[
            'id', 'image_path', 'pathologies', 'timestamp', 'patient_id', 'notes'
        ])

        # Try to load existing index and metadata
        self.load()

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        pathologies: List[List[str]],
        patient_ids: Optional[List[str]] = None,
        notes: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add embeddings to the vector store with metadata

        Args:
            embeddings: Array of embeddings (num_samples, embedding_dim)
            image_paths: List of image file paths
            pathologies: List of pathology labels for each image
            patient_ids: Optional list of patient IDs
            notes: Optional notes for each image

        Returns:
            List of assigned IDs for the added embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype('float32')

        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        end_id = self.index.ntotal

        ids = list(range(start_id, end_id))

        # Add metadata
        num_samples = len(image_paths)
        if patient_ids is None:
            patient_ids = [f"patient_{i}" for i in ids]
        if notes is None:
            notes = [""] * num_samples

        # Convert pathologies list to string
        pathologies_str = [",".join(p) if p else "No Finding" for p in pathologies]

        new_metadata = pd.DataFrame({
            'id': ids,
            'image_path': image_paths,
            'pathologies': pathologies_str,
            'timestamp': [datetime.now().isoformat()] * num_samples,
            'patient_id': patient_ids,
            'notes': notes
        })

        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)

        print(f"Added {num_samples} embeddings to the vector store")
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = config.K_NEIGHBORS
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Search for k nearest neighbors of the query embedding

        Args:
            query_embedding: Query embedding vector (1D array)
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (distances, indices, metadata_df)
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add embeddings first.")

        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')

        # Search in FAISS index
        k = min(k, self.index.ntotal)  # Ensure k doesn't exceed number of vectors
        distances, indices = self.index.search(query_embedding, k)

        # Get metadata for the results
        result_metadata = self.metadata.iloc[indices[0]].copy()
        result_metadata['distance'] = distances[0]
        result_metadata['similarity'] = 1 / (1 + distances[0])  # Convert distance to similarity

        return distances[0], indices[0], result_metadata

    def get_by_id(self, embedding_id: int) -> Dict:
        """
        Retrieve metadata for a specific embedding ID

        Args:
            embedding_id: The ID of the embedding

        Returns:
            Dictionary containing metadata
        """
        row = self.metadata[self.metadata['id'] == embedding_id]
        if row.empty:
            raise ValueError(f"No embedding found with ID {embedding_id}")
        return row.iloc[0].to_dict()

    def delete_by_id(self, embedding_id: int):
        """
        Remove an embedding from the store (marks as deleted in metadata)

        Note: FAISS doesn't support deletion, so we mark it in metadata
        To truly remove, rebuild the index without deleted items
        """
        self.metadata = self.metadata[self.metadata['id'] != embedding_id]
        print(f"Marked embedding {embedding_id} as deleted")

    def get_statistics(self) -> Dict:
        """
        Get statistics about the vector store

        Returns:
            Dictionary with statistics
        """
        total_embeddings = self.index.ntotal
        unique_patients = self.metadata['patient_id'].nunique()

        # Count pathologies
        all_pathologies = []
        for pathology_str in self.metadata['pathologies']:
            if pathology_str and pathology_str != "No Finding":
                all_pathologies.extend(pathology_str.split(','))

        pathology_counts = pd.Series(all_pathologies).value_counts().to_dict()

        return {
            'total_embeddings': total_embeddings,
            'unique_patients': unique_patients,
            'pathology_distribution': pathology_counts,
            'embedding_dimension': self.embedding_dim
        }

    def save(self):
        """
        Save the FAISS index and metadata to disk
        """
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        self.metadata.to_csv(self.metadata_path, index=False)

        print(f"Vector store saved to {self.index_path} and {self.metadata_path}")

    def load(self):
        """
        Load the FAISS index and metadata from disk
        """
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded FAISS index from {self.index_path}")

            if os.path.exists(self.metadata_path):
                self.metadata = pd.read_csv(self.metadata_path)
                print(f"Loaded metadata from {self.metadata_path}")
                print(f"Vector store contains {len(self.metadata)} embeddings")
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            print("Starting with empty vector store")

    def rebuild_index(self, embeddings_array: np.ndarray):
        """
        Rebuild the FAISS index from scratch (useful after deletions)

        Args:
            embeddings_array: Array of all embeddings (num_samples, embedding_dim)
        """
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        embeddings_array = embeddings_array.astype('float32')
        self.index.add(embeddings_array)
        print(f"Rebuilt FAISS index with {self.index.ntotal} embeddings")

    def clear(self):
        """
        Clear all data from the vector store
        """
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = pd.DataFrame(columns=[
            'id', 'image_path', 'pathologies', 'timestamp', 'patient_id', 'notes'
        ])
        print("Vector store cleared")


if __name__ == "__main__":
    # Example usage
    print("Testing VectorStore...")

    store = VectorStore()
    print(f"Vector store initialized with dimension {store.embedding_dim}")

    # Create dummy embeddings for testing
    dummy_embeddings = np.random.randn(5, config.EMBEDDING_DIM).astype('float32')
    dummy_paths = [f"image_{i}.png" for i in range(5)]
    dummy_pathologies = [
        ["Pneumonia"],
        ["Atelectasis", "Effusion"],
        ["No Finding"],
        ["Cardiomegaly"],
        ["Pneumothorax"]
    ]

    # Add embeddings
    ids = store.add_embeddings(dummy_embeddings, dummy_paths, dummy_pathologies)
    print(f"Added embeddings with IDs: {ids}")

    # Get statistics
    stats = store.get_statistics()
    print(f"\nVector Store Statistics:")
    print(stats)
