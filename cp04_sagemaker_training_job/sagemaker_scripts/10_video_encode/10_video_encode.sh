mv /root/yolox_tiny.onnx ./yolox_tiny.onnx
mv 10_video_encode/*.py .
yt-dlp -o $SM_HP_VIDEO_PATH -f 'bv[ext=mp4]+ba[ext=m4a]' $SM_HP_YOUTUBE_URL -q
df -h
python yolox_encode.py
ffmpeg -i ${SM_HP_OUTPUT_DIR}/out_video.mp4 -i $SM_HP_VIDEO_PATH -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 ${SM_HP_OUTPUT_DIR}/youtube.mp4
rm ${SM_HP_OUTPUT_DIR}/out_video.mp4