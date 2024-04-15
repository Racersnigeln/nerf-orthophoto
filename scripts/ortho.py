# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

# from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

import torch
import cv2
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (scale_gradients_by_distance_squared)

# from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE


def load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, load_step

def ortho_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "inference",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)

    # disable training mode in the network
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step

def orthophoto(nerfacto):
    def ortho_rays(direction, resolution):
        print("STARTING ORTHOPHOTO GENERATION")
        # Generate orthographic rays
        directions = torch.tensor([direction] * resolution * resolution, device="cuda").reshape(resolution, resolution, 3)
        origins = torch.zeros_like(directions)

        v1 = torch.ones(3, device="cuda")
        direction = torch.tensor(direction, device="cuda")
        v1 -= v1.dot(direction) * direction # make it orthogonal to the direction
        v1 /= torch.norm(v1)
        v2 = torch.cross(direction, v1) # another orthogonal vector
        v2 /= torch.norm(v2)

        for h in range(resolution):
            for w in range(resolution):
                point = torch.tensor([0.0, 0.0, 0.0], device="cuda")
                point += -1.0 + 2.0 * v1 * w / resolution
                point += -1.0 + 2.0 * v2 * w / resolution
                # if point.norm() > 1.0:
                #     point = torch.zeros(3)
                origins[h, w, :] = point
        return origins, directions
    
    # Generate orthographic rays
    direction = [-0.71132, 0.688117, -0.0043513]
    resolution = 500
    origins, directions = ortho_rays(direction, resolution)
    ray_bundle = RayBundle(origins=origins.cpu(), 
                           directions=directions.cpu(), 
                           pixel_area=torch.tensor(1).view(1, 1).cpu(), 
                           nears=torch.tensor(0.05).view(1, 1).cpu(), 
                           fars=torch.tensor(100).view(1, 1).cpu())
    print("RAY BUNDLE GENERATED")

    # Render the RGB image
    if nerfacto.training:
        nerfacto.camera_optimizer.apply_to_raybundle(ray_bundle)
    print("a")
    ray_samples: RaySamples
    print("b")
    ray_samples, weights_list, ray_samples_list = nerfacto.proposal_sampler(ray_bundle, density_fns=nerfacto.density_fns)
    print("c")
    field_outputs = nerfacto.field.forward(ray_samples, compute_normals=nerfacto.config.predict_normals)
    print("RAYS SAMPLED")
    if nerfacto.config.use_gradient_scaling:
        field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

    weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
    weights_list.append(weights)
    ray_samples_list.append(ray_samples)

    rgb = nerfacto.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    print("IMAGE RENDERED")

    # Convert RGB image to BGR format and save it
    bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("~/Pictures/orthophoto.jpg", bgr_image)
    print("IMAGE SAVED")

@dataclass
class RenderOrtho:
    """Load a checkpoint, render an orthophoto, and save it to a jpg file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("outputs/orthophoto.jpg")

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = ortho_setup(self.load_config, test_mode = "inference")
        # print(pipeline)
        # print("HEJ")
        nerf = pipeline.model
        orthophoto(nerf)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to

        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderOrtho).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderOrtho)  # noqa
