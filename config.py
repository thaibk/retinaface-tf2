# FOR DATALOADER
txt_path = "/mnt/sda/datasets/Face_Recognition/WIDER/train/label.txt"  ## Train txt path
min_sizes = [[16, 32], [64, 128], [256, 512]]
steps = [8, 16, 32]
variance = [0.1, 0.2]
image_shape = (640, 640, 3)
rgb_means = (104, 117, 123)

# FOR TRAIN
epoch_start = 0
batch_size = 1
loc_weight = 2.0
num_epochs = 10