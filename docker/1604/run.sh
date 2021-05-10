docker run  -it  -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix \
                       --device /dev/video0 \
                       -v /home/prasanna/repos:/host/repos/ slam_image 
