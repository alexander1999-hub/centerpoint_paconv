docker exec -it --user "docker_centerpoint" centerpoint \
    /bin/bash -c "source /opt/ros/noetic/setup.bash;
    cd /home/docker_centerpoint/catkin_ws;
    sudo -i;
    export CUDA_VISIBLE_DEVICES='0,1';
    export num_gpus=2;
    export PYTHONPATH="${PYTHONPATH}:/home/docker_centerpoint/catkin_ws/src/centerpoint/CenterPoint";
    export PYTHONPATH="${PYTHONPATH}:/home/docker_centerpoint/catkin_ws/src/centerpoint";
    export PYTHONPATH="${PYTHONPATH}:/home/docker_centerpoint/catkin_ws/src/centerpoint/nuscenes-devkit/python-sdk";
    /bin/bash"