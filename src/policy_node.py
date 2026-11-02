"""同時刻のROS画像・関節状態からπ0の指令列を生成する"""

import argparse

from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from gym_aloha.constants import JOINTS
import message_filters
import numpy as np
import rclpy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from policy import connect, predict
from ros_units import to_policy, to_ros


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, choices=range(1, 51), default=50,
                        help="予測した50点のうち実行する点数（1～50）")
    args = parser.parse_args()
    client, bridge = connect(), CvBridge()
    # 初回コンパイルを先に済ませる．このダミー指令はロボットには送らない．
    predict(client, np.zeros(14), np.zeros((480, 640, 3), dtype=np.uint8))
    rclpy.init()
    node = rclpy.create_node("aloha_policy")
    publisher = node.create_publisher(JointTrajectory, "/aloha/action_chunk", 1)
    states = message_filters.Subscriber(node, JointState, "/aloha/joint_states", qos_profile=1)
    images = message_filters.Subscriber(node, Image, "/aloha/top/image", qos_profile=1)
    synchronized = message_filters.TimeSynchronizer([states, images], queue_size=5)

    def infer(state, image):
        # 推論中にこのノードが待っても，別プロセスの物理シミュレーションは進む
        frame = bridge.imgmsg_to_cv2(image, desired_encoding="rgb8")
        actions = to_ros(predict(client, to_policy(state.position), frame, horizon=args.chunk_size))
        chunk = JointTrajectory(header=state.header, joint_names=JOINTS)
        for index, action in enumerate(actions):
            chunk.points.append(JointTrajectoryPoint(
                positions=action.tolist(), time_from_start=Duration(
                    sec=(index + 1) // 50, nanosec=((index + 1) % 50) * 20_000_000)))
        # ヘッダーは観測時刻を引き継ぐ．受信後に先頭から20 ms間隔で実行する．
        publisher.publish(chunk)

    synchronized.registerCallback(infer)
    print("π0の準備が完了しました。sim_node.pyを起動してください。", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
