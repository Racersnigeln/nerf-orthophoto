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

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

import torch
import cv2
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


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
    resolution = 4
    origins, directions = ortho_rays(direction, resolution)
    ray_bundle = RayBundle(origins=origins.cpu(), 
                        directions=directions.cpu(), 
                        pixel_area=torch.tensor(1).view(1, 1).cpu(), 
                        nears=torch.tensor(0.05).view(1, 1).cpu(), 
                        fars=torch.tensor(100).view(1, 1).cpu())
    print("RAY BUNDLE GENERATED")

    # Render the RGB image
    # if nerfacto.training:
    #     nerfacto.camera_optimizer.apply_to_raybundle(ray_bundle)
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
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        # print(pipeline)
        # print("HEJ")
        # nerfacto = pipeline.model
        # orthophoto(nerfacto)

        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
