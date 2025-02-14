import os
import base64
import uuid
from typing import List, Tuple
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from langchain.schema import Document
import torch
from config import LOADED_IMAGES_LIMIT


class ImageCaptioner:
    """Generates captions for images using a pretrained model (BLIP)."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", logger=None):

        self.logger = logger
        self.logger.info(f"Loading image captioning model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.to("cuda")
            self.logger.info("Using CUDA for image captioning")
        else:
            self.logger.info("Using CPU for image captioning")

    def _caption_image(self, image_path: str) -> str:
        """Generates a caption for the given image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.logger.error(f"Error captioning image {image_path}: {e}")
            return ""

    def _encode_image(self, image_path: str) -> str:
        """Encodes an image in base16 format."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b16encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            return ""

    def index_image(self, image_path: str) -> Tuple[List[Tuple[str, str]], List[Document]]:
        """
        Processes a single image: generates a caption, encodes the image, and wraps the results in a Document.

        Returns:
            Tuple[List[Tuple[str, str]], List[Document]]: A tuple containing a list with image ID and encoding, and a list of Documents.
        """
        retrieved_contents = []
        documents = []

        try:
            description = self._caption_image(image_path)
            encoded_image = self._encode_image(image_path)
            idx = str(uuid.uuid4())

            doc = Document(
                page_content=description,
                metadata={
                    "id": idx,
                    "type": "image",
                    "source_file": os.path.abspath(image_path),
                    "encoded_image": encoded_image,
                }
            )

            # print(doc.page_content)

            retrieved_contents.append((idx, encoded_image))
            documents.append(doc)
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}")
            raise e

        return retrieved_contents, documents

    def index_images_in_directory(self, root_folder: str, limit: int = None) -> Tuple[List[Tuple[str, str]], List[Document]]:
        """
        Recursively processes all images in the given root folder (and its subdirectories) sequentially.
        Displays progress using tqdm.

        Parameters:
            root_folder (str): The directory containing image files.
            limit (int, optional): Maximum number of images to process.

        Returns:
            Tuple[List[Tuple[str, str]], List[Document]]:
                - A list of tuples with image IDs and their encoded content.
                - A list of Document objects containing image metadata.
        """
        try:
            all_retrieved_contents = []
            all_documents = []
            image_paths = []

            for dirpath, _, filenames in os.walk(root_folder):
                for filename in filenames:
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_paths.append(os.path.join(dirpath, filename))
                        if limit and len(image_paths) >= limit:
                            break
                if limit and len(image_paths) >= limit:
                    break

            self.logger.info(f"Processing {len(image_paths)} images...")
            for image_path in tqdm(image_paths, desc="Indexing images", unit="image"):
                retrieved_contents, documents = self.index_image(image_path)
                all_retrieved_contents.extend(retrieved_contents)
                all_documents.extend(documents)

        except Exception as e:
            self.logger.error(f"Error processing directory {root_folder}")
            raise e

        return all_retrieved_contents, all_documents