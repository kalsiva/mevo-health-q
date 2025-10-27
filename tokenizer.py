"""
Chest X-ray Image Tokenizer
Converts chest X-ray images into vector embeddings using deep learning models
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import timm
import numpy as np
from typing import Union, List
import config


class ChestXRayTokenizer:
    """
    Tokenizer for chest X-ray images that creates vector embeddings
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        embedding_dim: int = config.EMBEDDING_DIM,
        pretrained: bool = config.PRETRAINED,
        device: str = None
    ):
        """
        Initialize the tokenizer with a pre-trained model

        Args:
            model_name: Name of the model to use (densenet121, resnet50, vit_base_patch16_224)
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pre-trained weights
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing ChestXRayTokenizer with {model_name} on {self.device}")

        # Load the base model
        self.model = self._load_model(model_name, pretrained, embedding_dim)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_name: str, pretrained: bool, embedding_dim: int) -> nn.Module:
        """
        Load and modify the model for feature extraction

        Args:
            model_name: Name of the model architecture
            pretrained: Whether to use pre-trained weights
            embedding_dim: Output embedding dimension

        Returns:
            Modified model for feature extraction
        """
        if model_name == "densenet121":
            # Load DenseNet-121 (commonly used for medical imaging)
            base_model = models.densenet121(pretrained=pretrained)
            num_features = base_model.classifier.in_features
            # Replace classifier with embedding layer
            base_model.classifier = nn.Linear(num_features, embedding_dim)

        elif model_name == "resnet50":
            # Load ResNet-50
            base_model = models.resnet50(pretrained=pretrained)
            num_features = base_model.fc.in_features
            # Replace final layer with embedding layer
            base_model.fc = nn.Linear(num_features, embedding_dim)

        elif model_name.startswith("vit"):
            # Load Vision Transformer using timm
            base_model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=embedding_dim
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return base_model

    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for the model

        Args:
            image: Path to image file, PIL Image, or numpy array

        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure image is in RGB mode
        if image.mode != 'RGB' and image.mode != 'L':
            image = image.convert('L')

        # Apply transformations
        image_tensor = self.transform(image)

        return image_tensor.unsqueeze(0)  # Add batch dimension

    def encode(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode a single image into a vector embedding

        Args:
            image: Path to image file, PIL Image, or numpy array

        Returns:
            Vector embedding as numpy array
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image).to(self.device)

        # Generate embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)

            # Normalize the embedding (L2 normalization)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten()

    def encode_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Encode multiple images into vector embeddings

        Args:
            images: List of image paths, PIL Images, or numpy arrays

        Returns:
            Array of embeddings with shape (num_images, embedding_dim)
        """
        # Preprocess all images
        image_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(batch_tensor)

            # Normalize embeddings
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def save_model(self, path: str):
        """Save the model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    print("Testing ChestXRayTokenizer...")

    tokenizer = ChestXRayTokenizer()
    print(f"Tokenizer initialized successfully!")
    print(f"Model: {tokenizer.model_name}")
    print(f"Embedding dimension: {tokenizer.embedding_dim}")
    print(f"Device: {tokenizer.device}")
