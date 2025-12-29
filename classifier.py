import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import os
import torch.nn.functional as F

class ImageClassifier:
    def __init__(self, references_dir: str = "references"):
        self.device = torch.device("cpu") # Focusing on offline/minimal compute, CPU is safer default.
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)
        self.model.eval()
        self.model.to(self.device)
        
        # We need the embedding, not the final classification. 
        # MobileNetV3 classifier part has a 'dropout' and 'fc'. We can just use the backbone + pooling or hook into it.
        # Alternatively, we can just replace the classifier with Identity or remove the last layer.
        
        # Taking the features before the final classification layer.
        # Structure: features -> avgpool -> classifier
        # We'll use the output of the avgpool layer effectively.
        
        self.preprocess = weights.transforms()
        self.references_dir = references_dir
        self.reference_embeddings = {}
        self.load_references()

    def get_embedding(self, image: Image.Image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass through features
            x = self.model.features(image_tensor)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            # Normalize embedding for cosine similarity to work best
            x = F.normalize(x, p=2, dim=1)
        return x

    def load_references(self):
        print("Loading references...")
        self.reference_embeddings = {}
        if not os.path.exists(self.references_dir):
            os.makedirs(self.references_dir)
            return

        for label in os.listdir(self.references_dir):
            label_dir = os.path.join(self.references_dir, label)
            if os.path.isdir(label_dir):
                embeddings = []
                for img_file in os.listdir(label_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        try:
                            img_path = os.path.join(label_dir, img_file)
                            image = Image.open(img_path).convert("RGB")
                            emb = self.get_embedding(image)
                            embeddings.append(emb)
                        except Exception as e:
                            print(f"Error loading {img_file}: {e}")
                if embeddings:
                    self.reference_embeddings[label] = torch.cat(embeddings)
        print(f"Loaded references for {len(self.reference_embeddings)} classes.")

    def classify(self, image: Image.Image, threshold: float = 0.7):
        if not self.reference_embeddings:
            return {"class": "Unknown", "confidence": 0.0, "message": "No references available"}

        target_emb = self.get_embedding(image) # Shape: [1, 576]
        
        best_score = -1.0
        best_label = "Unknown"

        for label, ref_embs in self.reference_embeddings.items():
            # ref_embs shape: [N, 576]
            # target_emb shape: [1, 576]
            # Cosine similarity is dot product of normalized vectors
            similarities = torch.mm(target_emb, ref_embs.t()) # Shape [1, N]
            score = torch.max(similarities).item()
            
            if score > best_score:
                best_score = score
                best_label = label

        if best_score < threshold:
            return {"class": "Unknown", "confidence": round(best_score, 4)}
        
        return {"class": best_label, "confidence": round(best_score, 4)}
