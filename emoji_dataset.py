"""
This module
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import clip
from tqdm.auto import tqdm
from typing import Dict, Tuple, List
from utils import get_device


class ClipEmbed:
    """
    A model that produces text embeddings using the CLIP model from OpenAI.
    """

    def __init__(self, device: str):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model = self.model.eval()
        self.device = device

    def embed(self, text: str) -> torch.Tensor:
        """
        Converts an input text (str) into a tensor embedding (torch.Tensor).
        """
        with torch.inference_mode():
            text = clip.tokenize(text).to(self.device)
            text_emb = self.model.encode_text(text)[0].cpu()
        return text_emb


class TextEmbedder:
    """
    A Text Embedder object that loads in pre-cached data and is also able to process text data and save it
    to disk for later re-loading and fast retrival.
    """

    def __init__(self):
        self.loaded = None

    def load_processed(self, data_path: str) -> None:
        """
        Loads in pre-processed text-embeddings from disk at data_path.
        """
        self.loaded = torch.load(data_path)

    def save_processed(self, all_texts: List[str], save_path: str) -> None:
        """
        Processes input text data through a CLIP embedding model and saves the results to disk.
        """
        os.makedirs(save_path, exist_ok=True)  # Create the directory if needed
        text_embedder = ClipEmbed(device=get_device())
        all_texts = list(set(all_texts))  # Remove duplicates

        # Encode all the texts with the ClipEmbed model
        idx_mapping = {}  # Text to index
        text_embeddings = []  # The text embedding of each text string
        for i, text in tqdm(enumerate(all_texts)):  # TODO: Consider if this could be batched instead
            idx_mapping[text] = i
            text_embeddings.append(text_embedder.embed(text))
        text_embeddings = torch.stack(text_embeddings)  # Combine into 1 large torch.Tensor

        # Run PCA to perform dimensional reduction on the text embeddings
        data = text_embeddings.float().numpy()
        mean = np.mean(data, axis=0)  # Compute mean vector
        centered_data = data - mean
        U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
        components = Vt  # Store all components
        components = torch.from_numpy(components).float()
        mean = torch.from_numpy(mean).float()

        # Save the results to disk for later quick retrival and use
        #   - idx_mapping is a dictionary that maps text strings to integer indices
        #   - embs is a torch.Tensor of text embeddings, we get the one desired by using
        #     embs[idx_mapping[text]] to look up which one is associated with each text
        #   - pca_components is a torch.Tensor of PCA components run on the text embeddings which can be used
        #     to quickly apply a PCA transform
        #   - mean is a torch.Tensor of means associated with the PCA transform
        torch.save({"idx_mapping": idx_mapping, "embs": text_embeddings,
                    "pca_components": components, "mean": mean}, save_path)

    def embed(self, *args, text: str = None, emb: torch.Tensor = None, num_pca: int = None) -> torch.Tensor:
        """
        Accepts an input text (str) or an embedding vector (torch.Tensor) and outputs an embedding vector
        (torch.Tensor) that has been PCA transformed (if num_pca is not None).

        If a text (str) is provided, then the pre-cached associated CLIP embedding is loaded and PCA
        transformed. If an embedding vector is provided (torch.Tensor) then just the PCA transform is
        performed.
        """
        assert (text is None) ^ (emb is None)

        if emb is None:
            emb_idx = self.loaded["idx_mapping"][text]
            emb = self.loaded["embs"][emb_idx].float()

        if num_pca is not None:
            emb = self.encode_pca(emb, num_pca)

        return emb

    def encode_pca(self, emb: torch.Tensor, num_pca: int) -> torch.Tensor:
        """
        Takes in a text embedding vector (torch.Tensor) and transforms it using the pre-cached PCA components
        and means.
        """
        emb = emb - self.loaded["mean"]
        emb = self.loaded["pca_components"][:num_pca] @ emb
        return emb

    def decode_pca(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Re-constructs an embedding vector in its original latent space by reversing a PCA transform.
        """
        num_pca = emb.shape[0]
        emb = self.loaded["pca_components"][:num_pca].T @ emb
        emb = emb + self.loaded["mean"]
        return emb


class EmojiDataset(Dataset):
    """
    Dataset object that loads in the pre-cached Emoji data and makes it ready for use.
    """

    def __init__(self, image_size: int, img_data_path: str = "data/emoji_data.npz",
                 text_emb_path: str = "data/text_embeddings.pt", num_text_emb_pca: int = None):
        """
        Initialized the dataset object.

        :param image_size: Sets the size of the images (height and width) returned by this dataset object.
            The images are originally [64 x 64 x 3].
        :param img_data_path: A file path pointing to the image data saved as a .npz file relative to the
            location of this file e.g. data/emoji_data.npz.
        :param text_emb_path: A file path pointing to the text_embeddings as a .pt file relative to the
            location of this file e.g. data/text_embeddings.pt.
        :param num_text_emb_pca: The dimension of the text embedding to use after applying a PCA transform.
        """
        data = np.load(os.path.join(CURRENT_DIR, img_data_path), allow_pickle=True)
        self.data = [data[key].item() for key in data]

        self.transform = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
        self.num_text_emb_pca = num_text_emb_pca
        self.text_embedder = TextEmbedder()
        self.text_embedder.load_processed(os.path.join(CURRENT_DIR, text_emb_path))

    def random_model_kwargs(self, n: int):
        """
        Returns n random model_kwargs sampled from the data set.
        """
        idxs = np.random.choice(len(self), n)
        samples = [self.__getitem__(idx) for idx in idxs]
        imgs, model_kwargs = torch.utils.data.default_collate(samples)
        return model_kwargs

    def embed_new_text(self, text: str, clip_embed: ClipEmbed) -> torch.Tensor:
        """
        Creates a new text embedding for an input text (string) using a ClipEmbed object instance
        and returns a torch.Tensor. Applies a PCA transform if self.num_text_emb_pca is not None.
        """
        text_emb = clip_embed.embed(text).float().cpu()
        if self.num_text_emb_pca is not None:  # Perform dimensional reduction using PCA is specified
            text_emb = self.text_embedder.encode_pca(text_emb, self.num_text_emb_pca)
        return text_emb

    def __len__(self) -> int:
        """
        Returns the length of the data set i.e. how many unique image classes there are in total.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns a random instance from the image class at index idx.

        :param idx: An integer denoting which image class to sample from.
        :returns:
            - img: One image randomly selected from the image class as a torch.Tensor of shape (C, H, W)
                where H = W = image_size.
            - model_kwargs: A dictionary with a key "text" containing a randomly selected image annotations
                associated with this image class and a key "text_emb" with a length 512 embedding vector of
                that text passed through the text embeder.
        """
        imgs = self.data[idx]["images"]  # Images from this class
        texts = self.data[idx]["texts"]  # Texts associated with this class

        # Select a random image from available images associated with this class
        img_idx = np.random.choice(len(imgs))
        img = imgs[img_idx]  # Extract out one image

        # Apply image pre-processing operations
        img = Image.fromarray(img)
        img = self.transform(img)  # Resize, crop, cast to torch.Tensor

        # Select a random text annotation from those associated with this class as well
        text = np.random.choice(texts)
        text_emb = self.text_embedder.embed(text=text, num_pca=self.num_text_emb_pca)
        model_kwargs = {"text_emb": text_emb, "text": text}

        # Return the image as a torch.Tensor of size (C, H, W) and a dict with keys: text_emb and text
        return img, model_kwargs
