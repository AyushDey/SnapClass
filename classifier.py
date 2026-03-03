import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, UnidentifiedImageError
import threading
import hashlib
import numpy as np
from pathlib import Path
from utils import setup_logger
from sqlalchemy.orm import Session
from db_actions import DBActions

# Need to import BookletCategory to fetch the text name by ID
from models import BookletCategory

logger = setup_logger("snapclass.classifier")

class ImageClassifier:
    def __init__(self, session_factory, references_dir: str | Path = "references"):
        self.session_factory = session_factory
        self.device = torch.device("cpu")
        self.references_dir = Path(references_dir)
        
        # Search Index State
        self.reference_embeddings = {}
        self.search_matrix = None
        self.search_labels = []
        self.search_categories = []
        
        self._lock = threading.Lock()
        
        # We apply 3 augmentations. Each is run twice -> 6 augmented + 1 original = 7 images per upload
        self._augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2)
        ]

        self._init_model()
        self.load_references()

    # =========================================================================
    # Model & Inference
    # =========================================================================

    def _init_model(self):
        try:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            self.model.fc = nn.Identity() # Remove classification layer
            self.model.eval()
            self.model.to(self.device)
            self.preprocess = weights.transforms()
            logger.info("ResNet18 model loaded.")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise e

    def get_embeddings(self, images: list[Image.Image]) -> list[list[float]]:
        """Processes a batch of images through the model simultaneously."""
        if not images:
            return []
        try:
            # Preprocess and stack into a single batch tensor: (Batch_Size, Channels, Height, Width)
            tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            with torch.no_grad():
                embs = F.normalize(self.model(tensors), p=2, dim=1)
            return embs.cpu().tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}")
            raise e

    def get_embedding(self, image: Image.Image) -> list[float]:
        """Convenience method for a single image."""
        return self.get_embeddings([image])[0]

    def classify(self, image: Image.Image, threshold: float = 0.70):
        """Classifies an image by comparing its embedding against the loaded search matrix."""
        with self._lock:
            if self.search_matrix is None:
                return {"class": "Unknown", "confidence": 0.0, "message": "No references available"}
            matrix = self.search_matrix
            labels = self.search_labels
            categories = self.search_categories
            db_session: Session = self.session_factory()

        try:
            # Multi-scale matching
            scores = self._compute_multi_scale_scores(image, matrix, labels, categories)
            
            # Sort by score descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            
            if not sorted_scores:
                 return {'class': 'Unknown Image'}
                 
            best_lbl, (best_score, best_category) = sorted_scores[0]
            result_class = best_lbl if best_score >= threshold else "Unknown"
            
            # Find other matches in the same category
            matches = [
                {"class": k, "score": round(v, 4)} 
                for k, (v, c) in sorted_scores 
                if k != result_class and c == best_category
            ][:5]
            db_actions = DBActions(db_session)
            # Fetch the actual category name from the database using the ID
            category_name = db_actions.get_category_by_id(best_category)
            
            if result_class != 'Unknown':
                response = {
                    "class": result_class,
                    "category_name": category_name,
                    "confidence": round(best_score, 4),
                    "matches": matches
                }
            else:
                response = {'class': 'Unknown', "confidence": 0.0, "message": "No references available"}
                
            return response
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise e

    def _compute_multi_scale_scores(self, image: Image.Image, matrix: torch.Tensor, labels: list, categories: list) -> dict:
        """Generates embeddings for original, 0.8x, and 1.2x scales simultaneously."""
        scores = {}
        images_to_embed = []
        
        # Prepare the 3 scale variants
        for scale in [1.0, 0.8, 1.2]:
            if abs(scale - 1.0) < 1e-6:
                images_to_embed.append(image)
            else:
                w, h = image.size
                images_to_embed.append(image.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS))
        
        # Run all 3 scales through the model in one batch
        embs_list = self.get_embeddings(images_to_embed)
        embs_tensor = torch.tensor(embs_list, dtype=torch.float32).to(self.device)
        
        # Compare all 3 embeddings against the database matrix
        for emb in embs_tensor:
            sims = torch.mm(emb.unsqueeze(0), matrix.t())
            vals, idxs = torch.topk(sims, k=min(100, sims.size(1)))
            
            for v, i in zip(vals[0].tolist(), idxs[0].tolist()):
                lbl = labels[i]
                cat = categories[i]
                # Keep the highest score seen for this label across all scales
                if v > scores.get(lbl, (-1.0, None))[0]:
                    scores[lbl] = (v, cat)
                    
        return scores

    # =========================================================================
    # File System Operations
    # =========================================================================

    def _compute_hash(self, path: str | Path) -> str:
        """Computes SHA256 hash of a file."""
        sha = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception as e:
            logger.error(f"Hash error {path}: {e}")
            return ""

    def _scan_local_references(self, manual_updates=None) -> dict:
        """
        Scans the reference directory for images.
        Returns: dict[file_hash] -> {path, label, category}
        """
        active_files = {}
        self.references_dir.mkdir(parents=True, exist_ok=True)
        
        for item_dir in self.references_dir.iterdir():
            if not item_dir.is_dir():
                continue
            
            contains_images = any(
                f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp') 
                for f in item_dir.iterdir()
            )
            
            if contains_images:
                label = item_dir.name
                category = "Uncategorized"
                for file_path in item_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                        if h := self._compute_hash(file_path):
                            active_files[h] = {"path": str(file_path), "label": label, "category": category}
            else:
                category = item_dir.name
                for label_dir in item_dir.iterdir():
                    if not label_dir.is_dir():
                        continue
                    label = label_dir.name
                    for file_path in label_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                            if h := self._compute_hash(file_path):
                                active_files[h] = {"path": str(file_path), "label": label, "category": category}

        if manual_updates:
            for h, info in manual_updates.items():
                if h in active_files:
                    active_files[h].update(info)
                else:
                    active_files[h] = info
                    
        return active_files

    # =========================================================================
    # Database Synchronization
    # =========================================================================

    def load_references(self, manual_updates=None):
        """Main entry point: Syncs disk -> DB -> Memory."""
        with self._lock:
            db_session: Session = self.session_factory()
            try:
                db_actions = DBActions(db_session)
                
                active_files = self._scan_local_references(manual_updates)
                active_hashes = set(active_files.keys())

                if not active_hashes:
                    self._clear_memory()
                    return

                self._sync_new_references(db_actions, active_files, active_hashes)
                self._prune_and_load_references(db_actions, active_files, active_hashes)

            except Exception as e:
                logger.exception(f"Database error during load_references: {e}")
                db_session.rollback()
            finally:
                db_session.close()

    def _sync_new_references(self, db_actions: DBActions, active_files: dict, active_hashes: set):
        """Identifies missing hashes and inserts embeddings + augmentations efficiently using batches."""
        existing_hashes = db_actions.get_existing_hashes()
        missing_hashes = active_hashes - existing_hashes

        if not missing_hashes:
            return

        logger.info(f"Preparing to compute embeddings for {len(missing_hashes)} new source files...")
        
        all_images = []
        metadata = [] # Stores (hash, label, category_id, variant_index)
        new_items = []
        batch_size = 32 # Process 32 images at once for maximum speed
        
        # 1. Open all missing images and generate their visual variants
        for h in missing_hashes:
            info = active_files[h]
            cat_name = info.get("category")
            cat_id = db_actions.get_or_create_category(cat_name)
            
            try:
                img = Image.open(info["path"]).convert("RGB")
                # Original + Augmentations (Total 7 variants per image)
                images = [img] + [t(img) for _ in range(2) for t in self._augmentations]
                
                # Flatten them into a master list for batched processing
                for idx, image in enumerate(images):
                    all_images.append(image)
                    metadata.append((h, info["label"], cat_id, idx))
            except Exception as e:
                logger.error(f"Error reading {info['path']}: {e}")

        if not all_images:
            return

        logger.info(f"Processing {len(all_images)} total image variants in batches of {batch_size}...")

        # 2. Run the images through the neural network in large chunks
        for i in range(0, len(all_images), batch_size):
            batch_imgs = all_images[i : i + batch_size]
            batch_meta = metadata[i : i + batch_size]
            
            try:
                embeddings = self.get_embeddings(batch_imgs)
                
                # Attach the results to the metadata for insertion
                for (h, label, cat_id, idx), emb in zip(batch_meta, embeddings):
                    new_items.append({
                        "image_hash": f"{h}_{idx}",
                        "item_name": label,
                        "category_id": cat_id,
                        "embedding": emb
                    })
            except Exception as e:
                logger.error(f"Error processing batch: {e}")

        # 3. Insert into database
        db_actions.insert_items(new_items)
        logger.info(f"Successfully inserted {len(new_items)} new embeddings.")

    def _prune_and_load_references(self, db_actions: DBActions, active_files: dict, active_hashes: set):
        """
        Iterates DB items. Keeps items that match active disk hashes, 
        deletes stale ones, and loads the active matrix into Memory.
        """
        db_items = db_actions.get_all_items()
        
        valid_items = []
        del_ids = []

        for item in db_items:
            base_hash = "_".join(item.image_hash.split("_")[:-1])
            
            if base_hash in active_hashes:
                # Sync category if changed
                active_cat_name = active_files[base_hash].get("category")
                if active_cat_name:
                    active_cat_id = db_actions.get_or_create_category(active_cat_name)
                    if item.category_id != active_cat_id:
                        item.category_id = active_cat_id
                
                valid_items.append(item)
            else:
                del_ids.append(item.id)

        if del_ids:
            db_actions.delete_items(del_ids)
            logger.info(f"Cleaned up {len(del_ids)} stale embeddings.")
        
        db_actions.commit()
        
        self._build_search_index(valid_items)

    def _build_search_index(self, items: list):
        """Converts a list of DB items into PyTorch tensors for searching."""
        if not items:
            self._clear_memory()
            return

        embeddings = [item.embedding for item in items]
        labels = [item.item_name for item in items]
        categories = [item.category_id for item in items]

        matrix = torch.tensor(np.array(embeddings), dtype=torch.float32).to(self.device)
        self.search_matrix = F.normalize(matrix, p=2, dim=1)
        self.search_labels = labels
        self.search_categories = categories
        
        self.reference_embeddings = {}
        for idx, lbl in enumerate(labels):
            if lbl not in self.reference_embeddings:
                 self.reference_embeddings[lbl] = []
            self.reference_embeddings[lbl].append(self.search_matrix[idx])
        
        logger.info(f"Loaded {len(labels)} embeddings for {len(set(labels))} classes into Search Matrix.")

    def _clear_memory(self):
        self.search_matrix = None
        self.search_labels = []
        self.search_categories = []
        self.reference_embeddings = {}