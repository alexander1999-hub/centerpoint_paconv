#!/bin/bash

cd "$(dirname "$0")"
cd ..
cd ..
cd ..
workspace_dir=$PWD

desktop_start() {
    docker run -itd --rm \
    --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    --net host \
    --ipc host \
    --privileged \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $workspace_dir/:/home/docker_centerpoint/catkin_ws:rw \
    --name centerpoint \
    ${ARCH}noetic/centerpoint:latest
xhost -

    # docker run -it -d --rm \
    #     --gpus '"device=0,1"' \
    #     --env="DISPLAY=$DISPLAY" \
    #     --env="QT_X11_NO_MITSHM=1" \
    #     --privileged \
    #     --name centerpoint \
    #     --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    #     --env="DISPLAY" \
    #     --net=host \
    #     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    #     -v $workspace_dir/:/home/docker_centerpoint/catkin_ws:rw \
    #     ${ARCH}noetic/centerpoint:latest
}

arm_start() {
    docker run -it -d --rm \
        --runtime nvidia \
        --runtime=runc \
        --interactive \
        --name centerpoint \
        --network host \
        --env=DISPLAY=$DISPLAY \
        --env=QT_X11_NO_MITSHM=1 \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw sshipway/xclock \
        --privileged \
        -v $workspace_dir/:/home/docker_centerpoint/catkin_ws:rw \
        ${ARCH}noetic/centerpoint:latest
}

main () {
    if [ "$(docker ps -aq -f status=exited -f name=centerpoint)" ]; then
        docker rm centerpoint;
    fi

    ARCH="$(uname -m)"

    if [ "$ARCH" = "x86_64" ]; then 
        desktop_start;
    elif [ "$ARCH" = "aarch64" ]; then
        arm_start;
    fi
    docker exec -it --user root centerpoint \
        /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/home/docker_centerpoint/catkin_ws/src/centerpoint/CenterPoint\";
        cd /home/docker_centerpoint/catkin_ws/src/centerpoint/CenterPoint;
        bash setup.sh;"
}

main;
    