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
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["state"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            # actions_32 = transforms.pad_to_dim(data["actions"], 32)
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ExcavatorOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 4 dims.
        return {"actions": np.asarray(data["actions"][:, :4])}



# @dataclasses.dataclass(frozen=True)
# class ExcavatorInputs(transforms.DataTransformFn):
#     """Transform excavator data using single camera to model input format.

#     """
#     model_type: _model.ModelType

#     def __call__(self, data: dict) -> dict:
#         def _parse_image(image):
#             image = np.asarray(image)
#             if np.issubdtype(image.dtype, np.floating):
#                 image = (255 * image).astype(np.uint8)
#             if image.ndim == 3 and image.shape[0] == 3:
#                 image = einops.rearrange(image, "c h w -> h w c")
#             return image

#         # 从 ConvertTensorToNumpy 包装后的字典中取图像
#         if "image" in data:
#             img_data = data["image"]
#             if isinstance(img_data, dict):
#                 main_image = _parse_image(next(iter(img_data.values())))
#             else:
#                 main_image = _parse_image(img_data)
#         else:
#             main_image = None

#         state = np.asarray(data["state"]) if "state" in data else None
#         actions = np.asarray(data["actions"]) if "actions" in data else None
        

#         # 只用一个摄像头，其他键用零数组填充
#         match self.model_type:
#             case _model.ModelType.PI0 | _model.ModelType.PI05:
#                 # Pi0/Pi05 期望 3 个摄像头，用零填充其他两个
#                 names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
#                 if main_image is not None:
#                     zero_img = np.zeros_like(main_image)
#                     images = (main_image, zero_img, zero_img)
#                     image_masks = (np.True_, np.False_, np.False_)
#                 else:
#                     zero_img = np.zeros((224, 224, 3), dtype=np.uint8)
#                     images = (zero_img, zero_img, zero_img)
#                     image_masks = (np.False_, np.False_, np.False_)
                    
#             case _model.ModelType.PI0_FAST:
#                 # Pi0-FAST 期望 3 个摄像头，用零填充其他两个
#                 names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
#                 if main_image is not None:
#                     zero_img = np.zeros_like(main_image)
#                     images = (main_image, zero_img, zero_img)
#                     image_masks = (np.True_, np.True_, np.True_)
#                 else:
#                     zero_img = np.zeros((224, 224, 3), dtype=np.uint8)
#                     images = (zero_img, zero_img, zero_img)
#                     image_masks = (np.False_, np.False_, np.False_)
#             case _:
#                 raise ValueError(f"Unsupported model type: {self.model_type}")

#         inputs = {
#             "state": state,
#             "image": dict(zip(names, images, strict=True)),
#             "image_mask": dict(zip(names, image_masks, strict=True)),
#         }

#         if actions is not None:
#             inputs["actions"] = actions

#         if "prompt" in data:
#             prompt = data["prompt"]
#             if isinstance(prompt, bytes):
#                 prompt = prompt.decode("utf-8")
#             inputs["prompt"] = prompt

#         return inputs
    


# @dataclasses.dataclass(frozen=True)
# class ExcavatorOutputs(transforms.DataTransformFn):
#     """Extract excavator actions - only return the first 4 dims."""
#     def __call__(self, data: dict) -> dict:
#         # Only return the first 4 dims to match excavator action_dim=4
#         return {"action": np.asarray(data["action"][:, :4])}