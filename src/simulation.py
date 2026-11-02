"""公開ALOHAの物理モデルを使い，必要なときだけカメラを描画する"""

from dm_control import mujoco
from dm_control.rl import control
from gym_aloha.constants import ASSETS_DIR, DT
from gym_aloha.tasks.sim import BOX_POSE, TransferCubeTask
from gym_aloha.utils import sample_box_pose


class CubeTask(TransferCubeTask):
    def get_observation(self, physics):
        # 元の処理は毎回3カメラを描画するため，画像取得を別の関数に分ける
        return {}


def make_env(seed=0, cube_xy=None):
    physics = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"))
    BOX_POSE[0] = sample_box_pose(seed)
    if cube_xy is not None:
        BOX_POSE[0][:2] = cube_xy
    env = control.Environment(physics, CubeTask(), control_timestep=DT)
    env.reset()
    return env


def observe(env):
    # 学習データと同じ上方カメラ・解像度．関節と画像を同じ時刻に取得する．
    state = env.task.get_qpos(env.physics)
    image = env.physics.render(height=480, width=640, camera_id="top")
    return state, image


def finger_contacts(env):
    # キューブに触れている指を集める．接触ペアの並び順には依存しない．
    physics = env.physics
    cube = physics.model.name2id("red_box", "geom")
    touching = set()
    for contact in physics.data.contact:
        if contact.geom1 == cube:
            touching.add(contact.geom2)
        if contact.geom2 == cube:
            touching.add(contact.geom1)
    return [all(physics.model.name2id(f"vx300s_{arm}/10_{side}_gripper_finger", "geom")
                in touching for side in ("left", "right")) for arm in ("left", "right")]
