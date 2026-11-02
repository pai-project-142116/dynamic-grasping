# 提出リポジトリのルートで source env.sh を実行する
export PAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$PAI_ROOT/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OPENPI_DATA_HOME="$XDG_CACHE_HOME/openpi"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$PAI_ROOT/.local/python"
export TMPDIR="$PAI_ROOT/outputs/tmp"
export ROS_HOME="$PAI_ROOT/outputs/ros"
export ROS_LOG_DIR="$ROS_HOME/log"
mkdir -p "$TMPDIR" "$ROS_LOG_DIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 他の実験のROSグラフと分け，同じPC内だけで通信する
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PAI_ROOT/ros_udp.xml"
