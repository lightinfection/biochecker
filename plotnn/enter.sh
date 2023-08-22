sudo xhost +local:
sudo docker run -it --rm --network host --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp/.X11-unix  --env="DISPLAY=$DISPLAY" plotnn /bin/bash
