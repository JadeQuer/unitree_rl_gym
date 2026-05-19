import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np


def convert_package_mesh_paths(src_urdf: Path, dst_urdf: Path) -> None:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    mesh_dir = src_urdf.parent / "meshes"
    dst_mesh_dir = dst_urdf.parent

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith("package://"):
            mesh_name = Path(filename.replace("package://", "", 1)).name
            mesh.attrib["filename"] = mesh_name
            src_mesh = mesh_dir / mesh_name
            if src_mesh.exists():
                dst_urdf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_mesh, dst_mesh_dir / mesh_name)

    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return None
    return data.xpos[body_id].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and render the A2D_dh URDF in MuJoCo.")
    parser.add_argument(
        "--urdf",
        default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf",
        help="Path to the source A2D URDF.",
    )
    parser.add_argument(
        "--converted",
        default="/root/autodl-tmp/unitree_rl_gym/reports/a2d_mujoco/A2D_mujoco.urdf",
        help="Output path for the MuJoCo-compatible URDF copy.",
    )
    parser.add_argument(
        "--image",
        default="/root/autodl-tmp/unitree_rl_gym/reports/a2d_mujoco/a2d_loaded.png",
        help="Output path for the rendered PNG.",
    )
    args = parser.parse_args()

    src_urdf = Path(args.urdf).expanduser().resolve()
    converted_urdf = Path(args.converted).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()

    convert_package_mesh_paths(src_urdf, converted_urdf)

    model = mujoco.MjModel.from_xml_path(str(converted_urdf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_types = {}
    for jnt_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        joint_type = int(model.jnt_type[jnt_id])
        joint_types[joint_type] = joint_types.get(joint_type, 0) + 1
        print(f"joint {jnt_id:02d}: {joint_name}")

    print(f"loaded: {converted_urdf}")
    print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom}")
    print(f"joint_type_counts={joint_types}")

    for name in ["base_link", "link_pitch_body", "Link7_r", "right_finger_base"]:
        pos = body_position(model, data, name)
        if pos is not None:
            print(f"{name}_pos={np.array2string(pos, precision=4)}")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array([0.0, 0.0, 0.6])
    camera.distance = 3.0
    camera.azimuth = 135
    camera.elevation = -20
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(image_path, image)
    print(f"image: {image_path}")


if __name__ == "__main__":
    main()
