import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, UnidentifiedImageError
import os
import torch.nn.functional as F
import torch.nn as nn
import threading
import hashlib
import chromadb
import numpy as np
from utils import setup_logger

# Initialize logger
logger = setup_logger("snapclass.classifier")

class ImageClassifier:
    def __init__(self, references_dir: str = "references", chroma_db_path: str = "./chroma_db"):
        self.device = torch.device("cpu") # Focusing on offline/minimal compute, CPU is safer default.
        logger.info(f"Initializing ImageClassifier on device: {self.device}")
        
        try:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            
            # Replace the final classification layer with Identity to get embeddings
            self.model.fc = nn.Identity()
            
            self.model.eval()
            self.model.to(self.device)
            self.preprocess = weights.transforms()
            logger.info("ResNet18 model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load user model: {e}")
            raise e

        self.references_dir = references_dir
        self.reference_embeddings = {}
        self._lock = threading.Lock() # Ensure thread safety for reference updates

        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="reference_embeddings",
                metadata={"hnsw:space": "cosine"} # Use cosine similarity
            )
            logger.info(f"Connected to ChromaDB at {chroma_db_path}")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB: {e}")
            raise e
        
        self.load_references()

    def _compute_image_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of the image file to detect changes/duplicates."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def get_embedding(self, image: Image.Image):
        try:
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # For ResNet18 with Identity fc:
                x = self.model(image_tensor)
                
                # Normalize embedding for cosine similarity to work best
                x = F.normalize(x, p=2, dim=1)
            return x
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise e

    def _process_image(self, img_path: str, augment_transforms: list) -> list:
        """
        Process a single image: load, classify (embed), and augment.
        Returns a list of embeddings (original + augmented).
        """
        embeddings = []
        try:
            # Validate image by opening
            original_image = Image.open(img_path).convert("RGB")
            
            # 1. Original
            emb = self.get_embedding(original_image)
            embeddings.append(emb)
            
            # 2. Augmented versions
            for _ in range(2): # Repeat random transforms a few times
                for t in augment_transforms:
                    aug_img = t(original_image)
                    aug_emb = self.get_embedding(aug_img)
                    embeddings.append(aug_emb)
                    
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Skipping corrupt or invalid image {os.path.basename(img_path)}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading {os.path.basename(img_path)}: {e}")
            
        return embeddings

    def _process_label_directory(self, label_dir: str, label: str, augment_transforms: list):
        """
        Process a specific label directory, iterating over all valid images.
        Returns a list of all embeddings collected for this label.
        """
        label_embeddings = []
        
        # Iterate over files in the label directory
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img_path = os.path.join(label_dir, img_file)
                # Process the individual image
                img_embeddings = self._process_image(img_path, augment_transforms)
                label_embeddings.extend(img_embeddings)
                
        return label_embeddings

    def _scan_disk_references(self) -> dict:
        """
        Scans the references directory and returns a map of {file_hash: (file_path, label)}.
        """
        active_files_map = {}
        
        if not os.path.exists(self.references_dir):
            logger.warning(f"References directory '{self.references_dir}' does not exist. Creating it.")
            os.makedirs(self.references_dir)
            return active_files_map

        for label in os.listdir(self.references_dir):
            label_dir = os.path.join(self.references_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for img_file in os.listdir(label_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    continue
                    
                img_path = os.path.join(label_dir, img_file)
                file_hash = self._compute_image_hash(img_path)
                
                if file_hash:
                    active_files_map[file_hash] = {"path": img_path, "label": label}
                    
        return active_files_map

    def _sync_new_files_to_db(self, active_files_map: dict) -> int:
        """
        Checks if active files are in DB, adds them if missing.
        Returns the count of new embeddings added.
        """
        new_embeddings_count = 0
        
        # Augmentation transforms for robustness
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ColorJitter(saturation=0.2)
        ]

        for file_hash, info in active_files_map.items():
            # Check if this file revision is already in DB
            existing = self.collection.get(
                ids=[f"{file_hash}_0"],
                include=[] 
            )
            
            if len(existing['ids']) == 0:
                # Missing in DB, compute and add
                img_path = info["path"]
                label = info["label"]
                
                emb_list = self._process_image(img_path, augment_transforms)
                
                if emb_list:
                    ids = [f"{file_hash}_{i}" for i in range(len(emb_list))]
                    # Convert tensor embeddings to list of floats for Chroma
                    embeddings_data = [e.cpu().tolist()[0] for e in emb_list]
                    metadatas = [{"label": label, "file_path": img_path, "hash": file_hash, "type": "original" if i==0 else "augmented"} for i in range(len(emb_list))]
                    
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings_data,
                        metadatas=metadatas
                    )
                    new_embeddings_count += len(emb_list)
                    logger.info(f"Computed and stored {len(emb_list)} embeddings for new/modified file: {os.path.basename(img_path)}")
        
        return new_embeddings_count

    def _cleanup_and_load_from_db(self, active_hashes: set) -> int:
        """
        Removes stale entries from DB and loads all valid embeddings into memory.
        Returns the count of restored (loaded) embeddings.
        """
        all_data = self.collection.get(include=['metadatas', 'embeddings'])
        
        if not all_data['ids']:
             logger.info("No data in ChromeDB.")
             return 0

        ids_to_delete = []
        loaded_embeddings = []
        loaded_labels = []
        
        for i, metadata in enumerate(all_data['metadatas']):
            if metadata['hash'] not in active_hashes:
                ids_to_delete.append(all_data['ids'][i])
            else:
                # Load into memory
                loaded_embeddings.append(all_data['embeddings'][i])
                loaded_labels.append(metadata['label'])
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} stale embeddings from DB.")
        
        if not loaded_embeddings:
            logger.info("No references available in DB after cleanup.")
            return 0
            
        # Convert back to Tensor [N, 512]
        self.search_matrix = torch.tensor(np.array(loaded_embeddings), dtype=torch.float32).to(self.device).detach()
        # Ensure normalization
        self.search_matrix = F.normalize(self.search_matrix, p=2, dim=1) 
        self.search_labels = loaded_labels
        
        # Reconstruct legacy reference_embeddings map for API/Debug
        label_indices = {}
        for idx, lbl in enumerate(loaded_labels):
            if lbl not in label_indices:
                label_indices[lbl] = []
            label_indices[lbl].append(idx)
            
        for lbl, indices in label_indices.items():
            self.reference_embeddings[lbl] = self.search_matrix[indices]

        return len(loaded_embeddings)

    def load_references(self):
        with self._lock:
            logger.info("Syncing references with ChromaDB...")
            self.reference_embeddings = {}
            self.search_matrix = None
            self.search_labels = []

            # 1. Scan disk
            active_files_map = self._scan_disk_references()
            active_hashes = set(active_files_map.keys())

            # 2. Sync to DB
            new_embeddings_count = self._sync_new_files_to_db(active_files_map)

            # 3. Cleanup and Load
            restored_embeddings_count = self._cleanup_and_load_from_db(active_hashes)

            if restored_embeddings_count > 0:
                unique_labels = set(self.search_labels)
                logger.info(f"Loaded references for {len(unique_labels)} classes. Total embeddings: {len(self.search_labels)} (New: {new_embeddings_count}, Cached: {restored_embeddings_count - new_embeddings_count})")

    def _get_scaled_embedding(self, image: Image.Image, scale: float):
        """Helper to resize image if needed and get embedding."""
        if abs(scale - 1.0) < 1e-9:
             return self.get_embedding(image)
        
        base_width, base_height = image.size
        new_size = (int(base_width * scale), int(base_height * scale))
        resized_img = image.resize(new_size, Image.Resampling.LANCZOS)
        return self.get_embedding(resized_img)

    def _collect_matches_from_embedding(self, target_emb, search_matrix) -> list:
        """Finds top matches for a single embedding against the matrix."""
        # Vectorized Cosine Similarity
        similarities = torch.mm(target_emb, search_matrix.t())
        
        # Check top K matches
        k = min(100, similarities.size(1))
        top_scores, top_indices = torch.topk(similarities, k=k)
        
        return list(zip(top_scores[0].tolist(), top_indices[0].tolist()))

    def classify(self, image: Image.Image, threshold: float = 0.5):
        # Optimization: Vectorized search O(1) Python overhead vs O(N) previously
        
        # Multi-scale inference
        scales = [1.0, 0.8, 1.2]
        
        best_overall_score = -1.0
        best_overall_label = "Unknown"
        final_matches_map = {} # label -> max_score

        # Snapshot references to avoid holding lock during heavy computation
        with self._lock:
            if self.search_matrix is None:
                logger.warning("Classification attempted with no references loaded.")
                return {"class": "Unknown", "confidence": 0.0, "message": "No references available"}
            
            search_matrix = self.search_matrix # [Total_N, 512]
            search_labels = self.search_labels

        try:
            for scale in scales:
                target_emb = self._get_scaled_embedding(image, scale)
                matches = self._collect_matches_from_embedding(target_emb, search_matrix)
                
                for score, idx in matches:
                    label = search_labels[idx]
                    
                    if label not in final_matches_map or score > final_matches_map[label]:
                        final_matches_map[label] = score

                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_label = label

            # Format matches list
            if best_overall_score >= threshold:
                result_class = best_overall_label
            else:
                result_class = "Unknown"

            # Format matches list, excluding the identified class
            all_scores = [
                {"class": k, "score": round(v, 4)} 
                for k, v in final_matches_map.items() 
                if k != result_class
            ]
            all_scores.sort(key=lambda x: x["score"], reverse=True)

            result = {
                "class": result_class,
                "confidence": round(best_overall_score, 4),
                "matches": all_scores[:5] # Top 5 matches excluding the winner
            }
            
            logger.debug(f"Classified image as {result['class']} (Conf: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise e
