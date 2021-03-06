#!/bin/bash

yellow=`tput setaf 3`
green=`tput setaf 2`
violet=`tput setaf 5`
reset_color=`tput sgr0`

ARCH="$(uname -m)"

main () {
    pushd "$(dirname "$0")";
    if [ "$ARCH" = "x86_64" ]; then 
        file="./Dockerfile"
    elif [ "$ARCH" = "aarch64" ]; then
        file="./Dockerfile.xavier"
    else
        echo "There is no Dockerfile for ${yellow}${ARCH}${reset_color} arch"
    fi
    echo "Building image for ${yellow}${ARCH}${reset_color} arch. from Dockerfile: ${yellow}${file}${reset_color}"
    docker build -f ${file} . -t ${ARCH}noetic/centerpoint:latest;
    popd;
}

main "$@"; exit;
