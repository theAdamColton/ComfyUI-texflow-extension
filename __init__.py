import numpy as np
import torch
from PIL import Image
import os
import hashlib

import nodes
import node_helpers
import folder_paths
import server


class LoadTexflowDepthImage:
    CATEGORY = "texflow"
    RETURN_TYPES = ("IMAGE", "MASK", "TEXFLOW_METADATA")
    FUNCTION = "load_depth_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(s):
        depth_image_path = folder_paths.get_annotated_filepath(
            "texflow_depth_image.tiff"
        )
        occ_image_path = folder_paths.get_annotated_filepath(
            "texflow_occupancy_image.png"
        )

        def hash_im(im_path):
            if not os.path.isfile(im_path):
                return "-"
            m = hashlib.sha256()
            with open(depth_image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()

        hash = hash_im(depth_image_path)
        hash += hash_im(occ_image_path)

        occ_img = node_helpers.pillow(Image.open, occ_image_path)
        occ_text_metadata = occ_img.text
        render_id = occ_text_metadata["render_id"]

        hash += render_id

        return hash

    def load_depth_image(self):
        depth_image_path = folder_paths.get_annotated_filepath(
            "texflow_depth_image.tiff"
        )
        depth_img = node_helpers.pillow(Image.open, depth_image_path)
        depth_img = np.asarray(depth_img)
        depth_img = depth_img.astype(np.float32) / (2**16 - 1)
        depth_img = torch.from_numpy(depth_img)

        occ_image_path = folder_paths.get_annotated_filepath(
            "texflow_occupancy_image.png"
        )

        occ_img = node_helpers.pillow(Image.open, occ_image_path)
        occ_text_metadata = occ_img.text
        render_id = occ_text_metadata["render_id"]

        occ_img = np.asarray(occ_img)
        occ_img = occ_img.astype(np.float32)
        occ_img = torch.from_numpy(occ_img)
        texflow_metadata = {"render_id": render_id}

        depth_img = depth_img.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        occ_img = occ_img.unsqueeze(0)

        return depth_img, occ_img, texflow_metadata


class SaveTexflowImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    CATEGORY = "texflow"

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    DESCRIPTION = "Adds a special texflow id as a suffix before saving the input images to your ComfyUI output directory."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "texflow_metadata": (
                    "TEXFLOW_METADATA",
                    {
                        "tooltip": "Connect this to the texflow metadata returned by load texflow depth image"
                    },
                ),
            },
        }

    def save_images(self, images, texflow_metadata):
        render_id = texflow_metadata["render_id"]
        return nodes.SaveImage.save_images(
            self, images, filename_prefix=f"texflow_{render_id}"
        )


NODE_CLASS_MAPPINGS = {
    "Load Texflow Depth Image": LoadTexflowDepthImage,
    "Save Texflow Image": SaveTexflowImage,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
