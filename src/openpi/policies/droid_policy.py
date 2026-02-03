import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class ExcavatorInputs(transforms.DataTransformFn):
    """
    Transform for excavator state -> action prediction.

    Input data format (from LeRobot):
    - image: Camera observation (H, W, C) uint8
    - state: Joint positions [boom, arm, bucket, swing] (4,) float32
    - action: Joint actions [boom, arm, bucket, swing] (4,) float32

    Output format (for model):
    - state: Joint positions (4,) float32
    - image: Dict with camera views (for tokenization)
    - image_mask: Dict with camera availability masks
    - action: (optional) Actions for training
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse state - excavator has 4 joints: [boom, arm, bucket, swing]
        state = np.asarray(data.get("state", data.get("observation/joint_position")))
        if state.ndim == 0:
            state = state[np.newaxis]
        if state.shape[-1] != 4:
            raise ValueError(
                f"Expected 4-dimensional state, got {state.shape[-1]}. "
                f"State should be [boom, arm, bucket, swing]."
            )

        # Parse image - excavator uses single camera
        # Handle both (C, H, W) and (H, W, C) formats
        # Try multiple possible keys: "image", "main", "observation/image"
        image = None
        for key in ["image", "main", "observation/image"]:
            if key in data:
                image = _parse_image(data[key])
                break
        if image is None:
            raise ValueError(
                f"Could not find image key in data. Available keys: {list(data.keys())}"
            )

        # For excavator, we only have one camera (not like DROID's 2+ cameras)
        # But we need to match the Pi0/Pi0.5 input format which expects 3 cameras
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Pad to 3 cameras to match base model
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (image, np.zeros_like(image), np.zeros_like(image))
                image_masks = (np.True_, np.False_, np.False_)  # Only 1st camera available
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (image, np.zeros_like(image), np.zeros_like(image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Include actions for training
        if "action" in data:
            inputs["action"] = np.asarray(data["action"])

        # Include prompt if provided
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ExcavatorOutputs(transforms.DataTransformFn):
    """
    Transform excavator model output to action space.

    The model outputs 32-dim actions (Pi0.5 standard), but excavator only needs 4.
    This extracts and validates the first 4 dimensions.

    Output dimensions map to: [boom, arm, bucket, swing]
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data.get("action", data.get("actions")))

        # Extract first 4 dimensions for excavator joints
        # Validate that we have at least 4 dims
        if actions.shape[-1] < 4:
            raise ValueError(
                f"Model output has {actions.shape[-1]} dims, need at least 4 "
                f"for [boom, arm, bucket, swing]."
            )

        # Return only the 4 excavator dimensions
        excavator_actions = actions[..., :4]

        return {"action": excavator_actions}
