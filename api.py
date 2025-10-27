"""
FastAPI REST API for Chest X-ray Tokenizer System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import shutil
from datetime import datetime
import config
from tokenizer import ChestXRayTokenizer
from vector_store import VectorStore
from inference import PathologyInferenceEngine


# Initialize FastAPI app
app = FastAPI(
    title="Chest X-ray Tokenizer API",
    description="API for encoding chest X-ray images and inferring pathologies using nearest neighbor search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the inference engine (singleton)
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize the inference engine on startup
    """
    global inference_engine
    print("Initializing Chest X-ray Tokenizer System...")

    try:
        tokenizer = ChestXRayTokenizer()
        vector_store = VectorStore()
        inference_engine = PathologyInferenceEngine(tokenizer, vector_store)
        print("✓ System initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        raise


# Pydantic models for request/response
class PathologyInferenceResponse(BaseModel):
    pathologies: List[str]
    confidence_scores: dict
    num_neighbors_found: int
    average_similarity: float
    neighbors: Optional[List[dict]] = None


class ReferenceImageRequest(BaseModel):
    pathologies: List[str]
    patient_id: Optional[str] = None
    notes: Optional[str] = None


class StatisticsResponse(BaseModel):
    total_embeddings: int
    unique_patients: int
    pathology_distribution: dict
    embedding_dimension: int


# API Endpoints

@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": "Chest X-ray Tokenizer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "infer": "/infer",
            "add_reference": "/add_reference",
            "statistics": "/statistics"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": inference_engine.tokenizer.model_name if inference_engine else "not loaded",
        "device": inference_engine.tokenizer.device if inference_engine else "unknown"
    }


@app.post("/infer", response_model=PathologyInferenceResponse)
async def infer_pathologies(
    file: UploadFile = File(...),
    return_neighbors: bool = True,
    k_neighbors: Optional[int] = None
):
    """
    Infer pathologies from an uploaded chest X-ray image

    Args:
        file: Chest X-ray image file (PNG, JPG, DICOM)
        return_neighbors: Whether to return neighbor details
        k_neighbors: Number of neighbors to consider (optional)

    Returns:
        PathologyInferenceResponse with inferred pathologies and confidence scores
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    # Save uploaded file temporarily
    temp_path = os.path.join(config.CACHE_DIR, f"temp_{datetime.now().timestamp()}_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Override k_neighbors if provided
        if k_neighbors is not None:
            original_k = inference_engine.k_neighbors
            inference_engine.k_neighbors = k_neighbors

        # Perform inference
        result = inference_engine.infer_pathologies(
            image_path=temp_path,
            return_neighbors=return_neighbors
        )

        # Restore original k_neighbors
        if k_neighbors is not None:
            inference_engine.k_neighbors = original_k

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/add_reference")
async def add_reference_image(
    file: UploadFile = File(...),
    pathologies: str = Form(...),
    patient_id: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """
    Add a reference chest X-ray image to the database

    Args:
        file: Chest X-ray image file
        pathologies: Comma-separated list of pathologies (e.g., "Pneumonia,Effusion")
        patient_id: Optional patient identifier
        notes: Optional notes

    Returns:
        Success message with embedding ID
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    # Parse pathologies
    pathology_list = [p.strip() for p in pathologies.split(',') if p.strip()]

    if not pathology_list:
        raise HTTPException(status_code=400, detail="At least one pathology must be provided")

    # Save file to data directory
    file_path = os.path.join(config.DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to database
        embedding_id = inference_engine.add_reference_image(
            image_path=file_path,
            pathologies=pathology_list,
            patient_id=patient_id,
            notes=notes
        )

        # Save state
        inference_engine.save_state()

        return {
            "status": "success",
            "message": "Reference image added successfully",
            "embedding_id": embedding_id,
            "file_path": file_path,
            "pathologies": pathology_list
        }

    except Exception as e:
        # Clean up file if adding failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to add reference: {str(e)}")


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get statistics about the reference database

    Returns:
        StatisticsResponse with database statistics
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        stats = inference_engine.get_store_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.post("/encode")
async def encode_image(file: UploadFile = File(...)):
    """
    Encode a chest X-ray image to a vector embedding

    Args:
        file: Chest X-ray image file

    Returns:
        Vector embedding as a list of floats
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    temp_path = os.path.join(config.CACHE_DIR, f"temp_{datetime.now().timestamp()}_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate embedding
        embedding = inference_engine.tokenizer.encode(temp_path)

        return {
            "status": "success",
            "embedding_dimension": len(embedding),
            "embedding": embedding.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/clear_database")
async def clear_database():
    """
    Clear all data from the vector store (use with caution!)

    Returns:
        Success message
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        inference_engine.vector_store.clear()
        inference_engine.save_state()
        return {
            "status": "success",
            "message": "Database cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    print(f"Starting Chest X-ray Tokenizer API on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
