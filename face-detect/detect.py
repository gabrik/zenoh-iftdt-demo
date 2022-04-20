import cv2
import numpy as np
import json
from zenoh_flow.interfaces import Operator
import io
import copy
import base64
from PIL import Image
import time

class DetectState:
    def __init__(self, configuration):
        if configuration.get('classes') is None:
            raise ValueError("Missing classes file path in configuration")
        if configuration.get('net_cfg') is None:
            raise ValueError("Missing YOLO network configuration")
        if configuration.get('net_weights') is None:
            raise ValueError("Missing YOLO netwokr weights")

        self.outfile = "/tmp/face-detect.csv"
        if configuration.get('outfile') is not None:
            self.outfile = configuration.get('outfile')

        classes_file = configuration.get('classes') #'./detection/face_classes.txt'
        net_cfg = configuration.get('net_cfg') #'./detection/yolov3-face.cfg'
        net_weights = configuration.get('net_weights') #./detection/yolov3-wider_16000.weights'
        model, classes = load_yolo(classes_file, net_cfg, net_weights)

        self.model = model
        self.classes = classes
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.4
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),100]

        self.file = open(self.outfile, "w+")
        self.file.write("node,time_in,time_out,kind")
        self.file.flush()

    def detect_faces(self, frame):
        required_size=(224, 224)
        classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        faces = []
        for (classid, score, box) in zip(classes, scores, boxes):
            label = classes[classid]
            x, y, width, height = box
            face = frame[y:y + height, x:x + width]
            if score > self.CONFIDENCE_THRESHOLD:
                image = Image.fromarray(face)
                image = image.resize(required_size)
                face_array = np.asarray(image)
                faces.append(face_array)
                # output_file_path = './face-detected.jpg'
                # print(f"Saving file {output_file_path}")
                # cv2.imwrite(
                # output_file_path,
                # face_array)

        return faces

class FaceDetection(Operator):
    def initialize(self, configuration):
         return DetectState(configuration)

    def finalize(self, state):
        state.file.close()
        return None

    def input_rule(self, _ctx, _state, _tokens):
        return True


    def output_rule(self, _ctx, state, outputs, _deadline_miss = None):
        return outputs

    def run(self, _ctx, state, inputs):
        intime = time.time_ns()

        # Getting the inputs
        data = inputs.get('Image').get_data()
        frame = bytes_to_frame(data)

        faces = state.detect_faces(frame)
        encoded_faces = []
        for f in faces:
                encoded_faces.append(frame_to_bytes(f, state.encode_params))

        base64_faces = list(map(lambda f: base64.b64encode(f).decode('ascii'), encoded_faces))

        outputs = {}

        # Sending out something only if faces detected
        if len(encoded_faces) > 0:
            output = {'detected_faces': base64_faces}
            #print(f'Detected {len(base64_faces)} faces')
            outputs['Faces'] = bytes(json.dumps(output), 'utf-8')

        outtime = time.time_ns()
        state.file.write(f'face-detection,{intime},{outtime},operator')
        state.file.flush()

        return outputs

def register():
    return FaceDetection

## Helper function

def resize_face(face, required_size=(224, 224)):
    face = np.resize(face, required_size)
    return face

def bytes_to_frame(bytes):
    img = np.load(io.BytesIO(bytes), allow_pickle=True)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def frame_to_bytes(frame, encode_params):
    jpeg = cv2.imencode('.jpg', frame, encode_params)[1]
    buf = io.BytesIO()
    np.save(buf, jpeg, allow_pickle=True)
    return buf.getvalue()

def read_classes(filename):
    classes = list()
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            classes.append(line)
    return classes

def load_yolo(classes_file, net_cfg, net_weights):
    net = cv2.dnn.readNet(net_cfg, net_weights)
    classes = read_classes(classes_file)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(512, 512), scale=1/255, swapRB=True)
    return model, classes


# Main for fast debug
def main():
    import time
    vs = FaceDetection()

    config = {
    'net_cfg': './face-detect/yolov3-face.cfg',
    'net_weights': './face-detect/yolov3-wider_16000.weights',
    'classes': './face-detect/face_classes.txt',
    }

    state = vs.initialize(config)
    camera = cv2.VideoCapture(0)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),100]
    while True:
        _, img = camera.read()
        time.sleep(0.40)
        jpeg = cv2.imencode('.jpg', img, encode_params)[1]
        buf = io.BytesIO()
        np.save(buf, jpeg, allow_pickle=True)
        res = vs.run(None, state, {'Image': buf.getvalue()})
        print(f'Res: {res}')

if __name__=='__main__':
    main()