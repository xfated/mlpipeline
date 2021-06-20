import requests
import json
import numpy as np
import cv2
import base64

import time

class Yolov5s:
    """ Wrapper for loading and serving yolov5s model"""
    
    def __init__(self):
        self.model_size = 640
        self.device = 'cpu'
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def predict(self, img, url, conf_thres=0.25, iou_thres=0.45):
        orig_shape = img.shape
        img = self._preproc(img)

        start = time.time()

        # Request from tensorflow 
        # Image is (1, 3, 640, 640)
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": img.tolist()
        })
        headers = {"content-type": "application/json"}
        json_response = requests.post(url, data=data, headers=headers)
        preds = json.loads(json_response.text)['predictions'][0]
        
        # Prep res for nms
        preds = np.expand_dims(np.asarray(preds['output_0']),0)

        # nms
        preds = self.nms(preds, conf_thres, iou_thres)

        result = []
        for pred in preds:
            result.append(self._proc_preds(pred[0], orig_shape))

        time_taken = (time.time() - start)
        return result, time_taken

    def _proc_preds(self, pred, orig_shape, model_size = None):
        if model_size is None:
            model_size = self.model_size
        
        # get resize ratio
        h, w, c = orig_shape
        longside = max(h,w)
        ratio = longside / model_size
        
        pred = pred[0]
        # resize
        bbox = pred[:4]
        bbox = bbox * ratio
        
        # create dict for res
        res = dict()
        res['bbox'] = bbox.astype(int).tolist()
        res['class'] = self.classes[int(pred[5])]
        res['conf'] = pred[4].item()

        return res

    # preproc img source to img
    def _preproc(self, img):
        img = self._resize(img)
        # cv2.imwrite('test.jpg',img)
        img = img.astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, 0)
        return img

    # Encode image to base64 string
    def encode_base64(self, img):
        img = cv2.imencode('.jpg', img)[1]
        return base64.b64encode(img).decode()

    # Decode image from base64 string
    def decode_base64(self, img_base64):
        img_nparray = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        return cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)

    # resize to target shape while maintaining aspect ratio. 
    # input:  (h, w, c)
    # output: (target, target, c)
    def _resize(self, img, target=640):
        h, w, c = img.shape
        # Get ratio for resize
        longside = max(h,w)
        ratio = target/longside
        new_width = int(w*ratio)
        new_height = int(h*ratio)

        # Resize while maintaining aspect ratio
        new_img = np.zeros((target, target, c))
        resized = cv2.resize(img, (new_width,new_height))
        new_img[:new_height,:new_width, :] = resized
        return new_img

    '''
        Input size: (batch, num boxes, res [bbox*4, conf, classes])
    '''
    def nms(self, prediction, conf_thres=0.25, iou_thres=0.45):
        
        num_classes = prediction.shape[2] - 5 
        candidates = prediction[..., 4] > conf_thres

        # 1 output list for each input image
        output = []
        ''' for batched nms '''
        for xi, x in enumerate(prediction):
            ''' eg Shapes
            x (before) = 25200, 85

            xi = 1
            candidates = (1, 25200)
            x (after)= (6, 85)
            '''
            x = x[candidates[xi]]

            # conf = obj_conf * cls_conf
            x[:, 5:] *= x[:, 4:5]

            # Bounding boxes
            box = self._xywh2xyxy(x[:, :4])

            # Multi label as default 
            '''
            Get coordinates of all classes that have larger conf than threshold
            (returns non-zero elements)
            i = x coord
            j = y coord
            '''
            i, j = np.nonzero(x[:, 5:] > conf_thres)
            '''
            x (bef) = (6,85)
            box[i] = (6,4)
            x[i, j+5, None] = (6,1) --> offset by 5 to get class confidence position 
            j[:, None] = (6,1) --> get class index
            x (aft) = (6, 6) 

            x now holds (bbox, confidence, class)
            '''
            x = np.concatenate((box[i], x[i, j+5, None], j[:, None]), 1)
            # Check num of boxes
            n = x.shape[0]
            if not n:
                continue
                
            ''' Compute NMS '''
            # coordinates of bounding boxes
            start_x = x[:, 0]
            start_y = x[:, 1]
            end_x = x[:, 2]
            end_y = x[:, 3]

            # Confidence scores of bounding boxes
            score = x[:, 4]

            # Classes/category of bounding boxes
            cat = x[:, 5]
            
            # Picked bounding boxes
            picked_boxes = []

            # Compute areas of bounding boxes
            areas = (end_x - start_x + 1) * (end_y - start_y + 1)

            # Sort by confidence score of bounding boxes
            order = np.argsort(score)

            # Iterate bounding boxes
            while order.size > 0:
                # The index of largest confidence score
                index = order[-1]

                # Pick the bounding box with largest confidence score
                picked_boxes.append([x[index]])

                # Compute ordinates of intersection-over-union(IOU)
                x1 = np.maximum(start_x[index], start_x[order[:-1]])
                x2 = np.minimum(end_x[index], end_x[order[:-1]])
                y1 = np.maximum(start_y[index], start_y[order[:-1]])
                y2 = np.minimum(end_y[index], end_y[order[:-1]])

                # Compute areas of intersection-over-union
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                intersection = w * h

                # Compute the ratio between intersection and union
                ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

                left = np.where(ratio < iou_thres)
                order = order[left]

        output.append(picked_boxes)

        return np.array(output)
        
    def _xywh2xyxy(self, x):
        '''
        compute for batch 
        '''
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

if __name__ == "__main__":
    yolo5s = Yolov5s()
    preds = yolo5s.predict('cat.jpg')
    print(preds)
