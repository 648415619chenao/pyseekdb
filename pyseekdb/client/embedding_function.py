"""
Embedding function interface and implementations

This module provides the EmbeddingFunction protocol and default implementations
for converting text documents to vector embeddings.
"""
import os
import multiprocessing
from typing import List, Protocol, Union, runtime_checkable, Optional, TypeVar
from multiprocessing import Queue, Process

# Set Hugging Face mirror endpoint for better download speed in China
# Users can override this by setting HF_ENDPOINT environment variable
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Type variable for input types
D = TypeVar('D')

# Type aliases
Documents = Union[str, List[str]]
Embeddings = List[List[float]]
Embedding = List[float]


def _embedding_worker_process(request_queue: Queue, response_queue: Queue, model_name: str):
    """
    Worker process that loads the model and handles embedding requests.
    
    This function runs in a separate process to isolate the model loading
    and embedding generation from the main process.
    
    Args:
        request_queue: Queue for receiving embedding requests
        response_queue: Queue for sending embedding responses
        model_name: Name of the sentence-transformers model to load
    """
    # Import sentence_transformers in the subprocess
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        response_queue.put(("error", ImportError(
            "sentence-transformers is not installed. "
            "Please install it with: pip install sentence-transformers"
        )))
        return
    
    # Load the model in the subprocess
    try:
        model = SentenceTransformer(model_name)
        # Get dimension from model
        test_embedding = model.encode(["test"], convert_to_numpy=True)
        dimension = len(test_embedding[0])
        
        # Send dimension info back
        response_queue.put(("dimension", dimension))
    except Exception as e:
        response_queue.put(("error", e))
        return
    
    # Process requests
    import queue
    while True:
        try:
            # Use timeout to allow periodic checking for shutdown
            try:
                request = request_queue.get(timeout=1.0)
            except queue.Empty:
                # Timeout occurred, continue loop to check again
                continue
            
            if request == "STOP":
                break
            
            if isinstance(request, tuple) and request[0] == "encode":
                texts = request[1]
                try:
                    embeddings = model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    # Convert numpy arrays to lists
                    result = [embedding.tolist() for embedding in embeddings]
                    response_queue.put(("result", result))
                except Exception as e:
                    response_queue.put(("error", e))
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            # Only send error if queue is still valid
            try:
                response_queue.put(("error", e))
            except:
                break


@runtime_checkable
class EmbeddingFunction(Protocol[D]):
    """
    Protocol for embedding functions that convert documents to vectors.
    
    This is similar to Chroma's EmbeddingFunction interface.
    Implementations should convert text documents to vector embeddings.
    
    Example:
        >>> class MyEmbeddingFunction:
        ...     def __call__(self, input: Documents) -> Embeddings:
        ...         # Convert documents to embeddings
        ...         return [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        >>> 
        >>> ef = MyEmbeddingFunction()
        >>> embeddings = ef(["Hello", "World"])
    """
    
    def __call__(self, input: D) -> Embeddings:
        """
        Convert input documents to embeddings.
        
        Args:
            input: Documents to embed (can be a single string or list of strings)
            
        Returns:
            List of embedding vectors (list of floats)
        """
        ...


class DefaultEmbeddingFunction:
    """
    Default embedding function using sentence-transformers.
    
    Uses the 'all-MiniLM-L6-v2' model by default, which produces 384-dimensional embeddings.
    This is a lightweight, fast model suitable for general-purpose text embeddings.
    
    The model loading and embedding generation are performed in a separate subprocess
    to isolate memory usage and avoid loading the model in the main process.
    
    Example:
        >>> ef = DefaultEmbeddingFunction()
        >>> embeddings = ef(["Hello world", "How are you?"])
        >>> print(len(embeddings[0]))  # 384
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the default embedding function.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (384 dimensions).
        """
        self.model_name = model_name
        self._dimension = None
        self._request_queue: Optional[Queue] = None
        self._response_queue: Optional[Queue] = None
        self._worker_process: Optional[Process] = None
        self._initialized = False
    
    def _ensure_worker_started(self):
        """Start the worker process if not already started"""
        if self._initialized:
            return
        
        # Create queues for inter-process communication
        self._request_queue = Queue()
        self._response_queue = Queue()
        
        # Start the worker process as daemon so it exits when main process exits
        self._worker_process = Process(
            target=_embedding_worker_process,
            args=(self._request_queue, self._response_queue, self.model_name),
            daemon=True
        )
        self._worker_process.start()
        
        # Wait for dimension info or error
        try:
            response = self._response_queue.get(timeout=60)  # 60 second timeout
            if response[0] == "dimension":
                self._dimension = response[1]
            elif response[0] == "error":
                raise response[1]
            else:
                raise RuntimeError(f"Unexpected response from worker: {response}")
        except Exception as e:
            self._cleanup()
            raise
        
        self._initialized = True
    
    def close(self):
        """Explicitly close the embedding function and clean up worker process"""
        self._cleanup()
    
    def _cleanup(self):
        """Clean up the worker process"""
        if not self._initialized:
            return
        
        if self._worker_process is not None and self._worker_process.is_alive():
            # Try graceful shutdown first
            if self._request_queue is not None:
                try:
                    self._request_queue.put("STOP")
                except:
                    # Queue might be full or closed, try terminate
                    pass
            
            # Wait for process to exit
            self._worker_process.join(timeout=2)
            
            # Force terminate if still alive
            if self._worker_process.is_alive():
                try:
                    self._worker_process.terminate()
                    self._worker_process.join(timeout=2)
                except:
                    pass
            
            # Force kill if still alive
            if self._worker_process.is_alive():
                try:
                    self._worker_process.kill()
                    self._worker_process.join(timeout=1)
                except:
                    pass
        
        self._worker_process = None
        self._request_queue = None
        self._response_queue = None
        self._initialized = False
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function"""
        self._ensure_worker_started()
        if self._dimension is None:
            raise RuntimeError("Dimension not available")
        return self._dimension
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.
        
        Args:
            input: Single document (str) or list of documents (List[str])
            
        Returns:
            List of embedding vectors
            
        Example:
            >>> ef = DefaultEmbeddingFunction()
            >>> # Single document
            >>> embedding = ef("Hello world")
            >>> # Multiple documents
            >>> embeddings = ef(["Hello", "World"])
        """
        self._ensure_worker_started()
        
        # Handle single string input
        if isinstance(input, str):
            input = [input]
        
        # Handle empty input
        if not input:
            return []
        
        # Send encoding request to worker process
        self._request_queue.put(("encode", input))
        
        # Wait for response
        try:
            response = self._response_queue.get(timeout=300)  # 5 minute timeout
            if response[0] == "result":
                return response[1]
            elif response[0] == "error":
                raise response[1]
            else:
                raise RuntimeError(f"Unexpected response from worker: {response}")
        except Exception as e:
            raise RuntimeError(f"Error getting embedding from worker process: {e}") from e
    
    def __repr__(self) -> str:
        return f"DefaultEmbeddingFunction(model_name='{self.model_name}')"
    
    def __del__(self):
        """Cleanup on deletion"""
        self._cleanup()


# Global default embedding function instance
_default_embedding_function: Optional[DefaultEmbeddingFunction] = None


def get_default_embedding_function() -> DefaultEmbeddingFunction:
    """
    Get or create the default embedding function instance.
    
    Returns:
        DefaultEmbeddingFunction instance
    """
    global _default_embedding_function
    if _default_embedding_function is None:
        _default_embedding_function = DefaultEmbeddingFunction()
    return _default_embedding_function

