docker run --rm --gpus all -p 8500:8500 \
--mount type=bind,source=/home/mdt/OwnCloud/face_recognition/git/re_retinaface_tf2/saved_model,target=/models/RetinaFace \
-e MODEL_NAME=RetinaFace -t tensorflow/serving:latest-gpu &