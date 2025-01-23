from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

sam2_checkpoint = "/ssd/ml/sam2_baseline_weights/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda:0')

predictor = SAM2ImagePredictor(sam2_model)

image = Image.open('/ssd/ml/map-scale-v2/outputs/BB/SJC10_2-20200401-PA1766140_bb0.png')
image = np.array(image.convert("RGB"))

predictor.set_image(image)

input_point = np.array([[5000, 5500]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
# sorted_ind = np.argsort(scores)[::-1]
# masks = masks[sorted_ind]
# scores = scores[sorted_ind]
# logits = logits[sorted_ind]
