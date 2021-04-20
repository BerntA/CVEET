import time
import cv2
import numpy as np
import requests as req
import tensorflow as tf
from collections import defaultdict
from object_detection.utils import visualization_utils as viz_utils
from utils import show_image

def get_image(url, flags=cv2.IMREAD_COLOR):
    try:
        r = req.get(url)
        if r.status_code != 200: # Is the camera down?
            return None
        img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), flags)
        img = img[25:,:,:] # Crop away the top black bar.
        return img
    except:
        print("Response timed out!")
        return None

def get_detections(det, labels, thresh=0.4):
    res = defaultdict(list)
    for score, class_idx, box in zip(
        det['detection_scores'], 
        det['detection_classes'], 
        det['detection_boxes']
        ):
        if score < thresh:
            continue
        h = hash(' '.join(['{:.5f}'.format(v) for v in box.tolist()]))
        res[h].append((score, labels[class_idx].get('name')))

    for v in res.values():
        v.sort(key=lambda j:j[0], reverse=True)
    
    return [v[0][1] for v in res.values()]

def inference(model, image_np, input_tensor, labels, visualize_img=False, thresh=0.4):
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = { key: value[0,:num_detections].numpy() for key, value in detections.items() }
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    if visualize_img:
        start_time = time.time()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            labels,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=thresh,
            agnostic_mode=False,
            line_thickness=1,
            mask_alpha=0.4
        )        
        show_image(image_np, size=(16,12))
        print('Inference took {} seconds'.format(time.time()-start_time))
    else: # Return an array of predictions otherwise.
        return get_detections(detections, labels, thresh)
        
def inference_url(url, model, labels, visualize_img=False, thresh=0.4):
    img = get_image(url)
    if img is None:
        return
    inference(model, img, tf.convert_to_tensor(img), labels, visualize_img, thresh)
