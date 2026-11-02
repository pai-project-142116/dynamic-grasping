"""物理を50 Hz，カメラを10 Hzで進める．推論は別プロセスで行う．"""

import argparse
import json
from collections import deque
from pathlib import Path
import time

from cv_bridge import CvBridge
from gym_aloha.constants import DT, JOINTS
import imageio.v2 as imageio
import numpy as np
import rclpy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

from ros_units import to_policy, to_ros
from simulation import finger_contacts, make_env, observe
from block_motion import BlockMotion
from block_paths import configure, path
from correction import Correction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube-xy", nargs=2, type=float, help="キューブ初期位置のx y [m]")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--video", default="outputs/ros.mp4")
    parser.add_argument("--trajectory", help="ブロック軌道のJSON．省略すると静止")
    parser.add_argument("--correction", choices=["none", "position", "velocity", "mlp"],
                        help="補正方式。noneは両腕の連続化だけ。省略時は従来の直接実行")
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("aloha_sim")
    states = node.create_publisher(JointState, "/aloha/joint_states", 1)
    images = node.create_publisher(Image, "/aloha/top/image", 1)
    config = configure(json.loads(Path(args.trajectory).read_text())) if args.trajectory else None
    cube_xy = path(config, 0)[0] if config else args.cube_xy
    env, bridge = make_env(args.seed, cube_xy), CvBridge()
    motion = BlockMotion(config) if config else None
    correction = Correction(env, args.correction) if args.correction is not None else None
    actions, frames, ages, records = deque(), [], [], []
    contacts, positions = [], []
    released = []
    joint_states, joint_commands, base_actions, observed, image_times = [], [], [], [], []
    pending_reference = None
    action = env.task.get_qpos(env.physics)
    waiting = False

    def receive(message):
        nonlocal waiting
        # 指令列は最後まで使う．推論待ちの間は最後の関節目標を保持する．
        actions.extend(to_policy([point.positions for point in message.points]))
        if correction:
            correction.set_reference(pending_reference)
        waiting = False
        observed = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        age = node.get_clock().now().nanoseconds * 1e-9 - observed
        ages.append(age)

    node.create_subscription(JointTrajectory, "/aloha/action_chunk", receive, 1)
    # 最初の1枚がROSの接続完了前に失われないよう，相手の購読開始を待つ
    while images.get_subscription_count() == 0 or states.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    started, max_reward, max_height = time.monotonic(), 0, 0.0
    for step in range(round(args.seconds / DT)):
        rclpy.spin_once(node, timeout_sec=0)
        if actions:
            action = actions.popleft()
        if motion:
            motion.move(env)
        actual = env.task.get_qpos(env.physics).copy()
        command = correction.apply(action, actual, float(env.physics.data.time)) if correction else action
        joint_states.append(actual)
        joint_commands.append(command.copy())
        base_actions.append(action.copy())
        result = env.step(command)
        max_reward = max(max_reward, result.reward)
        max_height = max(max_height, float(env.physics.data.qpos[18]))
        records.append([float(env.physics.data.time), result.reward, float(env.physics.data.qpos[18])])
        contacts.append(finger_contacts(env))
        if motion:
            motion.check_grasp(contacts[-1], float(env.physics.data.time))
        released.append(motion.released if motion else False)
        positions.append(env.physics.data.qpos[16:19].copy())
        state = JointState(name=JOINTS, position=to_ros(env.task.get_qpos(env.physics)).tolist())
        # 実時間の取得時刻を共通ヘッダーに使う．use_sim_timeはfalseのままにする．
        state.header.stamp = node.get_clock().now().to_msg()
        states.publish(state)
        if step % 5 == 0:
            _, frame = observe(env)
            if correction:
                valid = correction.tracker.observe(frame, float(env.physics.data.time))
                observed.append(correction.tracker.latest.copy())
                image_times.append(float(env.physics.data.time))
            image = bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            image.header = state.header
            # 次の予測は指令列を使い切ってから1回だけ要求する
            if not actions and not waiting:
                if correction:
                    pending_reference = correction.tracker.latest.copy() if valid else None
                images.publish(image)
                waiting = True
            frames.append(frame)
        # 推論応答を待たず，単調時計を基準に20 msずつ進める
        time.sleep(max(0, started + (step + 1) * DT - time.monotonic()))
    elapsed = time.monotonic() - started
    print(f"sim={env.physics.data.time:.3f}s wall={elapsed:.3f}s max_reward={max_reward} "
          f"max_cube_z={max_height:.3f} chunks={len(ages)}", flush=True)
    if ages:
        print(f"observation_to_command_mean={sum(ages) / len(ages):.3f}s", flush=True)
    if motion:
        print(f"trajectory_release_time={motion.release_time}", flush=True)
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.video, frames, fps=10)
    np.savez(Path(args.video).with_suffix(".npz"), records=records, ages=ages,
             contacts=contacts, cube_positions=positions, released=released, wall_seconds=elapsed,
             joint_states=joint_states, joint_commands=joint_commands, base_actions=base_actions,
             observed=observed, image_times=image_times)
    if correction:
        correction.close()
    env.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
