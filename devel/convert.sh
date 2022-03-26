#!/bin/bash

# THIS SCRIPT CONVERTS EVERY MP4 (IN THE CURRENT FOLDER AND SUBFOLDER) TO A MULTI-BITRATE VIDEO IN MP4-DASH
# For each file "videoname.mp4" it creates a folder "dash_videoname" containing a dash manifest file "stream.mpd" and subfolders containing video segments.
# Explanation:
# https://rybakov.com/blog/

# Validation tool:
# http://dashif.org/conformance.html

# MDN reference:
# https://developer.mozilla.org/en-US/Apps/Fundamentals/Audio_and_video_delivery/Setting_up_adaptive_streaming_media_sources

# Add the following mime-types (uncommented) to .htaccess:
# AddType video/mp4 m4s
# AddType application/dash+xml mpd

# Use type="application/dash+xml"
# in html when using mp4 as fallback:
#                <video data-dashjs-player loop="true" >
#                    <source src="/walking/walking.mpd" type="application/dash+xml">
#                    <source src="/walking/walking.mp4" type="video/mp4">
#                </video>

# DASH.js
# https://github.com/Dash-Industry-Forum/dash.js

MYDIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
SAVEDIR=$(pwd)

# Check programs
if [ -z "$(which ffmpeg)" ]; then
    echo "Error: ffmpeg is not installed"
    exit 1
fi

if [ -z "$(which MP4Box)" ]; then
    echo "Error: MP4Box is not installed"
    exit 1
fi

cd "$MYDIR"

TARGET_FILES=$(find ./ -maxdepth 1 -type f \( -name "*.mov" -or -name "*.mp4" \))
for f in $TARGET_FILES
do
  fe=$(basename "$f") # fullname of the file
  f="${fe%.*}" # name without extension

  if [ ! -d "${f}" ]; then #if directory does not exist, convert
    echo "Converting \"$f\" to multi-bitrate video in MPEG-DASH"

    mkdir "${f}"

    ffmpeg -y -i "${fe}" -c:a aac -b:a 192k -vn "${f}_audio.m4a"

    ffmpeg -y -i "${fe}" -preset slow -tune film -vsync passthrough -an -c:v libx264 -x264opts 'keyint=25:min-keyint=25:no-scenecut' -crf 22 -maxrate 5000k -bufsize 12000k -pix_fmt yuv420p -f mp4 "${f}_5000.mp4"
    ffmpeg -y -i "${fe}" -preset slow -tune film -vsync passthrough -an -c:v libx264 -x264opts 'keyint=25:min-keyint=25:no-scenecut' -crf 23 -maxrate 3000k -bufsize 6000k -pix_fmt yuv420p -f mp4  "${f}_3000.mp4"
    ffmpeg -y -i "${fe}" -preset slow -tune film -vsync passthrough -an -c:v libx264 -x264opts 'keyint=25:min-keyint=25:no-scenecut' -crf 23 -maxrate 1500k -bufsize 3000k -pix_fmt yuv420p -f mp4   "${f}_1500.mp4"
    ffmpeg -y -i "${fe}" -preset slow -tune film -vsync passthrough -an -c:v libx264 -x264opts 'keyint=25:min-keyint=25:no-scenecut' -crf 23 -maxrate 800k -bufsize 2000k -pix_fmt yuv420p -vf "scale=-2:720" -f mp4  "${f}_800.mp4"
    ffmpeg -y -i "${fe}" -preset slow -tune film -vsync passthrough -an -c:v libx264 -x264opts 'keyint=25:min-keyint=25:no-scenecut' -crf 23 -maxrate 400k -bufsize 1000k -pix_fmt yuv420p -vf "scale=-2:540" -f mp4  "${f}_400.mp4"
    # static file for ios and old browsers and mobile safari
    ffmpeg -y -i "${fe}" -preset slow -tune film -movflags +faststart -vsync passthrough -c:a aac -b:a 160k -c:v libx264  -crf 23 -maxrate 2000k -bufsize 4000k -pix_fmt yuv420p -f mp4 "${f}/${f}.mp4"


    rm -f ffmpeg*log*
    # if audio stream does not exist, ignore it
    if [ -e "${f}_audio.m4a" ]; then
        MP4Box -dash 2000 -rap -frag-rap  -bs-switching no -profile "dashavc264:live" "${f}_5000.mp4" "${f}_3000.mp4" "${f}_1500.mp4" "${f}_800.mp4" "${f}_400.mp4" "${f}_audio.m4a" -out "${f}/${f}.mpd"
        rm "${f}_5000.mp4" "${f}_3000.mp4" "${f}_1500.mp4" "${f}_800.mp4" "${f}_400.mp4" "${f}_audio.m4a"
    else
        MP4Box -dash 2000 -rap -frag-rap  -bs-switching no -profile "dashavc264:live" "${f}_5000.mp4" "${f}_3000.mp4" "${f}_1500.mp4" "${f}_800.mp4" "${f}_400.mp4" -out "${f}/${f}.mpd"
        rm "${f}_5000.mp4" "${f}_3000.mp4" "${f}_1500.mp4" "${f}_800.mp4" "${f}_400.mp4"
    fi
    # create a jpg for poster. Use imagemagick or just save the frame directly from ffmpeg is you don't have cjpeg installed.
    ffmpeg -i "${fe}" -ss 00:00:00 -vframes 1  -qscale:v 10 -n -f image2 - | cjpeg -progressive -quality 75 -outfile "${f}"/"${f}".jpg

    fi

done

cd "$SAVEDIR"