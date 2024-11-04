import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image_name = "data/images/train2017/000000000034.jpg"
image = cv2.imread(image_name) # f'images/{image_name}.jpg'
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


sam_checkpoint = "out/training/epoch-002-f1_0.57-ckpt.pth" # epoch_10_200_img
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

print("Mask: ", masks)

print("Len Masks:", len(masks))
print("Masks keys:", masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig(f"images/inference_{image_name}_b.png")

# Tuning parameters
# mask_generator_2 = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )
# masks2 = mask_generator_2.generate(image)
# print("Len Masks2:", len(masks2))

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# show_anns(masks2)

# plt.savefig(f"images/inference_{image_name}_b_tuning_options.png")