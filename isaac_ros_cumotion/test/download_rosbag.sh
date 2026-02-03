# Download rosbag and metadata.

if [ -z "$ISAAC_ROS_WS" ]; then
    echo "ISAAC_ROS_WS is not set"
    exit 1
fi

ROSBAG_PATH=$ISAAC_ROS_WS/isaac_ros_assets/r2b_2024/r2b_robotarm

mkdir -p $ROSBAG_PATH
rm -rf $ROSBAG_PATH/*
cd $ROSBAG_PATH

wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/isaac/r2bdataset2024/1/files?redirect=true&path=r2b_robotarm/metadata.yaml' --output-document 'metadata.yaml'
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/isaac/r2bdataset2024/1/files?redirect=true&path=r2b_robotarm/r2b_robotarm_0.mcap' --output-document 'r2b_robotarm_0.mcap'
