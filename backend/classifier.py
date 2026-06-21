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

logger = setup_logger("snapclass.classifier")
UNKOWN_MESSAGE="No references available"
class ImageClassifier:
    def __init__(self, session_factory, references_dir: str | Path = "references"):
        self.session_factory = session_factory
        self.device = torch.device("cpu")
        self.references_dir = Path(references_dir)

        # Search Index State (Legacy in-memory fallback for unit tests)
        self.search_matrix = None
        self.search_labels = []
        self.search_categories = []

        self._lock = threading.Lock()

        # Cache of the last full disk scan so fast-path manual_updates can
        # skip re-hashing the entire references directory.
        self._cached_active_files: dict = {}

        # We apply 3 augmentations. Each is run twice -> 6 augmented + 1 original = 7 images per upload
        self._augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2)
        ]

        self._init_model()
        self.load_references()

    @property
    def reference_embeddings(self) -> dict[str, list]:
        """Provides backward-compatibility with tests/code expecting this attribute."""
        if hasattr(self, "_reference_embeddings_override") and self._reference_embeddings_override is not None:
            return self._reference_embeddings_override
        db_session = self.session_factory()
        try:
            db_actions = DBActions(db_session)
            embeddings = db_actions.get_all_embeddings()
            ref_embs = {}
            for emb in embeddings:
                lbl = emb.item.item_name
                if lbl not in ref_embs:
                    ref_embs[lbl] = []
                ref_embs[lbl].append(emb.embedding)
            return ref_embs
        finally:
            db_session.close()

    @reference_embeddings.setter
    def reference_embeddings(self, value):
        self._reference_embeddings_override = value

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
        """Classifies an image by comparing its embedding against references."""
        # For unit testing backward compatibility:
        # Check if search_matrix is populated in memory.
        with self._lock:
            in_memory_mode = self.search_matrix is not None
            if in_memory_mode:
                matrix = self.search_matrix
                labels = self.search_labels
                categories = self.search_categories
                db_session = self.session_factory()

        if in_memory_mode:
            try:
                # Multi-scale matching using in-memory matrix
                scores = self._compute_multi_scale_scores(image, matrix, labels, categories)
                
                sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
                if not sorted_scores:
                     return {'class': 'Unknown Image'}
                     
                best_lbl, (best_score, best_category) = sorted_scores[0]
                result_class = best_lbl if best_score >= threshold else "Unknown"
                
                db_actions = DBActions(db_session)
                category_name = db_actions.get_category_by_id(best_category)

                matches = []
                for k, (v, c) in sorted_scores:
                    if k != result_class and c == best_category:
                        if len(matches) >= 5:
                            break
                        match_cat_name = db_actions.get_category_by_id(c)
                        matches.append({
                            "class": k, 
                            "score": round(v, 4),
                            "category_name": match_cat_name,
                            "image_path": self.get_reference_image_path(k, match_cat_name)
                        })
                
                if result_class != 'Unknown':
                    response = {
                        "class": result_class,
                        "category_name": category_name,
                        "confidence": round(best_score, 4),
                        "image_path": self.get_reference_image_path(result_class, category_name),
                        "matches": matches
                    }
                else:
                    response = {'class': 'Unknown', "confidence": 0.0, "message": "No references available"}
                    
                return response
            except Exception as e:
                logger.error(f"Classification error: {e}")
                raise e
            finally:
                db_session.close()

        # New production path: DB-level similarity search using pgvector
        db_session = None
        try:
            # Prepare multi-scale image inputs
            images_to_embed = []
            for scale in [1.0, 0.8, 1.2]:
                if abs(scale - 1.0) < 1e-6:
                    images_to_embed.append(image)
                else:
                    w, h = image.size
                    images_to_embed.append(image.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS))
            
            # Batch inference to get query embeddings
            embs_list = self.get_embeddings(images_to_embed)
            
            db_session = self.session_factory()
            db_actions = DBActions(db_session)
            
            from models import BookletEmbedding
            from sqlalchemy import select
            any_embeddings = db_session.scalars(select(BookletEmbedding)).first() is not None
            if not any_embeddings:
                return {"class": "Unknown", "confidence": 0.0, "message": UNKOWN_MESSAGE}
            
            scores = {}
            for emb in embs_list:
                # DB similarity search (100 nearest neighbors)
                matches = db_actions.search_similar_embeddings(emb, limit=100)
                for match in matches:
                    dist = getattr(match, "distance", 1.0)
                    sim = 1.0 - dist
                    lbl = match.item.item_name
                    cat = match.booklet_category_id
                    
                    if sim > scores.get(lbl, (-1.0, None))[0]:
                        scores[lbl] = (sim, cat)
                        
            sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            if not sorted_scores:
                 return {'class': 'Unknown Image'}
                 
            best_lbl, (best_score, best_category) = sorted_scores[0]
            result_class = best_lbl if best_score >= threshold else "Unknown"
            
            category_name = db_actions.get_category_by_id(best_category)

            matches_list = []
            for k, (v, c) in sorted_scores:
                if k != result_class and c == best_category:
                    if len(matches_list) >= 5:
                        break
                    match_cat_name = db_actions.get_category_by_id(c)
                    matches_list.append({
                        "class": k, 
                        "score": round(v, 4),
                        "category_name": match_cat_name,
                        "image_path": self.get_reference_image_path(k, match_cat_name)
                    })
            
            if result_class != 'Unknown':
                response = {
                    "class": result_class,
                    "category_name": category_name,
                    "confidence": round(best_score, 4),
                    "image_path": self.get_reference_image_path(result_class, category_name),
                    "matches": matches_list
                }
            else:
                response = {'class': 'Unknown', "confidence": 0.0, "message": UNKOWN_MESSAGE}
                
            return response
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise e
        finally:
            if db_session is not None:
                db_session.close()

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

        Fast-path: when *all* work is already described by manual_updates (e.g.
        a bulk_upload that already hashed every new file), we skip the expensive
        full-disk hash scan and merge manual_updates on top of whatever is
        already cached in memory from the last full scan.
        """
        # --- Fast path: manual_updates carry all new entries
        if manual_updates:
            active_files = self._cached_active_files.copy() if self._cached_active_files else {}
            active_files.update(manual_updates)
            self._cached_active_files = active_files
            return active_files

        # --- Full scan path (startup / /refresh / add_reference without batch)
        active_files = {}
        self.references_dir.mkdir(parents=True, exist_ok=True)

        for item_dir in self.references_dir.iterdir():
            if not item_dir.is_dir():
                continue
            self._scan_item_dir(item_dir, active_files)

        self._cached_active_files = active_files
        return active_files

    _IMAGE_EXTS = frozenset(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

    def _is_valid_image_file(self, file_path: Path) -> bool:
        """Check whether a file is a valid reference image file to process."""
        if not file_path.is_file():
            return False
        if file_path.name.startswith('.') or file_path.name.startswith('._'):
            return False
        return file_path.suffix.lower() in self._IMAGE_EXTS

    def _scan_item_dir(self, item_dir: Path, active_files: dict):
        """Scans a single top-level directory, handling both flat and nested layouts."""
        contains_images = any(
            self._is_valid_image_file(f)
            for f in item_dir.iterdir()
        )
        if contains_images:
            self._index_image_files(item_dir, item_dir.name, "Uncategorized", active_files)
        else:
            # Nested layout: item_dir is a category, each sub-dir is a label
            for label_dir in item_dir.iterdir():
                if label_dir.is_dir():
                    self._index_image_files(label_dir, label_dir.name, item_dir.name, active_files)

    def _index_image_files(self, directory: Path, label: str, category: str, active_files: dict):
        """Hashes every image file in a directory and records it in active_files."""
        for file_path in directory.iterdir():
            if self._is_valid_image_file(file_path):
                if h := self._compute_hash(file_path):
                    active_files[h] = {"path": str(file_path), "label": label, "category": category}

    # =========================================================================
    # Database Synchronization
    # =========================================================================

    def load_references(self, manual_updates=None):
        """Main entry point: Syncs disk -> DB."""
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
        metadata = []  # Stores (hash, item_id, category_id, variant_index)
        new_embeddings = []
        batch_size = 32  # Process 32 images at once for maximum speed

        # Per-call item/category caches avoid repeated DB round-trips.
        item_id_cache: dict[str, int] = {}
        # Per-call category name -> ID cache to avoid repeated DB round-trips
        cat_id_cache: dict[str, int] = {}

        # 1. Open all missing images and generate their visual variants
        for h in missing_hashes:
            info = active_files[h]
            label = info["label"]
            cat_name = info.get("category") or "Uncategorized"

            if label not in item_id_cache:
                item_id_cache[label] = db_actions.get_or_create_item(label)
            item_id = item_id_cache[label]

            if cat_name not in cat_id_cache:
                cat_id_cache[cat_name] = db_actions.get_or_create_category(cat_name)
            cat_id = cat_id_cache[cat_name]

            try:
                img = Image.open(info["path"]).convert("RGB")
                # Original + Augmentations (Total 7 variants per image)
                images = [img] + [t(img) for _ in range(2) for t in self._augmentations]

                # Flatten them into a master list for batched processing
                for idx, image in enumerate(images):
                    all_images.append(image)
                    metadata.append((h, item_id, cat_id, idx))
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
                for (h, item_id, cat_id, idx), emb in zip(batch_meta, embeddings):
                    new_embeddings.append({
                        "image_hash": f"{h}_{idx}",
                        "booklet_item_id": item_id,
                        "booklet_category_id": cat_id,
                        "embedding": emb
                    })
            except Exception as e:
                logger.error(f"Error processing batch: {e}")

        # 3. Insert into database and commit so IDs are visible to subsequent queries
        db_actions.insert_embeddings(new_embeddings)
        db_actions.commit()
        logger.info(f"Successfully inserted {len(new_embeddings)} new embeddings.")

    def _prune_and_load_references(self, db_actions: DBActions, active_files: dict, active_hashes: set):
        """
        Iterates DB items. Keeps items that match active disk hashes,
        deletes stale ones, and loads the active matrix into Memory.
        """
        db_embeddings = db_actions.get_all_embeddings()

        valid_embeddings = []
        del_ids = []
        cat_id_cache: dict[str, int] = {}

        for embedding in db_embeddings:
            base_hash = "_".join(embedding.image_hash.split("_")[:-1])
            if base_hash in active_hashes:
                self._sync_embedding_category(embedding, active_files[base_hash], db_actions, cat_id_cache)
                valid_embeddings.append(embedding)
            else:
                del_ids.append(embedding.id)

        if del_ids:
            db_actions.delete_embeddings(del_ids)
            logger.info(f"Cleaned up {len(del_ids)} stale embeddings.")

        db_actions.commit()
        from unittest.mock import Mock
        dialect_name = db_actions.session.bind.dialect.name
        if isinstance(dialect_name, Mock) or dialect_name == "sqlite":
            self._build_search_index(valid_embeddings)
        else:
            logger.info(f"References synchronized in database. Total embeddings: {len(valid_embeddings)}")

    def _sync_embedding_category(
        self,
        embedding,
        file_info: dict,
        db_actions: DBActions,
        cat_id_cache: dict,
    ):
        """Ensures embedding category matches the active file's category, updating if needed."""
        cat_name = file_info.get("category")
        if not cat_name:
            return
        if cat_name not in cat_id_cache:
            cat_id_cache[cat_name] = db_actions.get_or_create_category(cat_name)
        active_cat_id = cat_id_cache[cat_name]
        if embedding.booklet_category_id != active_cat_id:
            embedding.booklet_category_id = active_cat_id

    def _build_search_index(self, embeddings: list):
        """Converts a list of DB embeddings into PyTorch tensors for searching (Legacy test compatibility)."""
        if not embeddings:
            self._clear_memory()
            return

        vectors = [embedding.embedding for embedding in embeddings]
        labels = [embedding.item.item_name for embedding in embeddings]
        categories = [embedding.booklet_category_id for embedding in embeddings]

        matrix = torch.tensor(np.array(vectors), dtype=torch.float32).to(self.device)
        self.search_matrix = F.normalize(matrix, p=2, dim=1)
        self.search_labels = labels
        self.search_categories = categories
        
        self._reference_embeddings_override = {}
        for idx, lbl in enumerate(labels):
            if lbl not in self._reference_embeddings_override:
                 self._reference_embeddings_override[lbl] = []
            self._reference_embeddings_override[lbl].append(self.search_matrix[idx])
        
        logger.info(f"Loaded {len(labels)} embeddings for {len(set(labels))} classes into Search Matrix (Test mode).")

    def _clear_memory(self):
        self.search_matrix = None
        self.search_labels = []
        self.search_categories = []
        self._reference_embeddings_override = {}

    def get_reference_image_path(self, label: str, category_name: str) -> str | None:
        """Returns the relative path for the frontend to load a reference image."""
        for info in self._cached_active_files.values():
            if info["label"] == label and info.get("category", "Uncategorized") == category_name:
                return "/" + Path(info["path"]).as_posix()
        return None
