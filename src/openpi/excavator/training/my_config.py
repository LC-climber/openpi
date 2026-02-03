from dataclasses import dataclass

from openpi.transforms import Group, RepackTransform
import openpi.transforms as T

from openpi.policies.droid_policy import DroidInputs, DroidOutputs
from openpi.models.pi0_config import Pi0Config
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.training.config import DataConfigFactory
from openpi.training.config import TrainConfig


@dataclass(frozen=True)
class LeRobotExcavatorDataConfig(DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack_transforms = Group(
            inputs=[
                RepackTransform(
                    {
                        "state": "observation/joint_position",
                        "action": "actions",
                    }
                )
            ]
        )

        data_transforms = Group(
            inputs=[
                DroidInputs(model_type=model_config.model_type),
                T.InjectDefaultPrompt("control the excavator joints"),
            ],
            outputs=[DroidOutputs()],
        )

        model_transforms = ModelTransformFactory(
            default_prompt="control the excavator joints"
        )(model_config)

        return DataConfig(
            repo_id=self.repo_id,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
            use_quantile_norm=True,
        )


def get_excavator_configs():
    from openpi.training.config import TrainConfig  # ← 延迟 import（关键！）

    return [
        TrainConfig(
            name="pi05_excavator_finetune",
            exp_name="excavator_lora_v1",
            model=Pi0Config(
                pi05=True,
                action_dim=4,
                action_horizon=1,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotExcavatorDataConfig(
                repo_id="your_hf_username/lerobot_excavator_data"
            ),
            weight_loader=CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            ema_decay=None,
            batch_size=32,
            num_train_steps=5000,
        )
    ]
