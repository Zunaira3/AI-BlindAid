
import speech_recognition as sr
import cv2
import depthai as dai
import time
import pyttsx3
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import numpy as np
import east
import blobconverter
# Initialize speech recognition engine and text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Initialize text-to-speech engine

def run_object_detection():
    import sys
    nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    if 1 < len(sys.argv):
        arg = sys.argv[1]
        if arg == "yolo3":
            nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
        elif arg == "yolo4":
            nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
        else:
            nnBlobPath = arg
    else:
        print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

    if not Path(nnBlobPath).exists():
        import sys
        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

    # Tiny yolo v3/4 label texts
    labelMap = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    ]

    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")
    nnNetworkOut.setStreamName("nnNetwork")

    # Properties
    camRgb.setPreviewSize(416, 416)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
    stereo.setSubpixel(True)

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
    spatialDetectionNetwork.setIouThreshold(0.5)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    if syncNN:
       spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
       camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
    spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        printOutputLayersOnce = True
        prevframe=None

        while True:

        
            inPreview = previewQueue.get()
            inDet = detectionNNQueue.get()
            depth = depthQueue.get()
            inNN = networkQueue.get()

            if printOutputLayersOnce:
                toPrint = 'Output layer names:'
                for ten in inNN.getAllLayerNames():
                   toPrint = f'{toPrint} {ten},'
                print(toPrint)
                printOutputLayersOnce = False;

            frame = inPreview.getCvFrame()
            depthFrame = depth.getFrame() # depthFrame values are in millimeters
            if prevframe is not None:
                diff=np.sum(cv2.absdiff(frame,prevframe))
                if diff>1500: 
                   continue
            prevframe=frame.copy()

            depth_downscaled = depthFrame[::4]
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            detections = inDet.detections

            #If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            for detection in detections:
                roiData = detection.boundingBoxMapping
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

                #Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x/1000)} m", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y/1000)} m", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z/1000)} m", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                #Speech output
                object_label = labelMap[detection.label]
                distance = int(detection.spatialCoordinates.z / 1000)
                speech_output =f"There is a {object_label} at {distance} meters."
                engine.say(speech_output)
                engine.runAndWait()

            #Show the frame
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.imshow("rgb", frame)
            cv2.imshow("depth", depthFrameColor)

            if voice_command=="stop":        
               return

def OCR():
    class HostSeqSync:
        def __init__(self):
            self.imfFrames = []
        def add_msg(self, msg):
            self.imfFrames.append(msg)
        def get_msg(self, target_seq):
            for i, imgFrame in enumerate(self.imfFrames):
                if target_seq == imgFrame.getSequenceNum():
                    self.imfFrames = self.imfFrames[i:]
                    break
            return self.imfFrames[0]

    pipeline = dai.Pipeline()
    version = "2021.2"
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

    colorCam = pipeline.create(dai.node.ColorCamera)
    colorCam.setPreviewSize(256, 256)
    colorCam.setVideoSize(1024, 1024) # 4 times larger in both axis
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
    colorCam.setFps(10)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')
    controlIn.out.link(colorCam.inputControl)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName('video')
    colorCam.video.link(cam_xout.input)

# ---------------------------------------
# 1st stage NN - text-detection
# ---------------------------------------

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blobconverter.from_zoo(name="east_text_detection_256x256",zoo_type="depthai",shaves=6, version=version))
    colorCam.preview.link(nn.input)

    nn_xout = pipeline.create(dai.node.XLinkOut)
    nn_xout.setStreamName('detections')
    nn.out.link(nn_xout.input)

# ---------------------------------------
# 2nd stage NN - text-recognition-0012
# ---------------------------------------

    manip = pipeline.create(dai.node.ImageManip)
    manip.setWaitForConfigInput(True)

    manip_img = pipeline.create(dai.node.XLinkIn)
    manip_img.setStreamName('manip_img')
    manip_img.out.link(manip.inputImage)

    manip_cfg = pipeline.create(dai.node.XLinkIn)
    manip_cfg.setStreamName('manip_cfg')
    manip_cfg.out.link(manip.inputConfig)

    manip_xout = pipeline.create(dai.node.XLinkOut)
    manip_xout.setStreamName('manip_out')

    nn2 = pipeline.create(dai.node.NeuralNetwork)
    nn2.setBlobPath(blobconverter.from_zoo(name="text-recognition-0012", shaves=6, version=version))
    nn2.setNumInferenceThreads(2)
    manip.out.link(nn2.input)
    manip.out.link(manip_xout.input)

    nn2_xout = pipeline.create(dai.node.XLinkOut)
    nn2_xout.setStreamName("recognitions")
    nn2.out.link(nn2_xout.input)

    def to_tensor_result(packet):
        return {
            name: np.array(packet.getLayerFp16(name))
            for name in [tensor.name for tensor in packet.getRaw().tensors]
        }

    def to_planar(frame):
        return frame.transpose(2, 0, 1).flatten()

    with dai.Device(pipeline) as device:
       q_vid = device.getOutputQueue("video", 4, blocking=False)
    # This should be set to block, but would get to some extreme queuing/latency!
       q_det = device.getOutputQueue("detections", 4, blocking=False)

       q_rec = device.getOutputQueue("recognitions", 4, blocking=True)

       q_manip_img = device.getInputQueue("manip_img")
       q_manip_cfg = device.getInputQueue("manip_cfg")
       q_manip_out = device.getOutputQueue("manip_out", 4, blocking=False)

       controlQueue = device.getInputQueue('control')

       frame = None
       cropped_stacked = None
       rotated_rectangles = []
       rec_pushed = 0
       rec_received = 0
       host_sync = HostSeqSync()

       class CTCCodec(object):
           """ Convert between text-label and text-index """
           def __init__(self, characters):
           # characters (str): set of the possible characters.
               dict_character = list(characters)

               self.dict = {}
               for i, char in enumerate(dict_character):
                   self.dict[char] = i + 1
               self.characters = dict_character
                #print(self.characters)
                #input()
           def decode(self, preds):
                """ convert text-index into text-label. """
                texts = []
                index = 0
                # Select max probabilty (greedy decoding) then decode index to character
                preds = preds.astype(np.float16)
                preds_index = np.argmax(preds, 2)
                preds_index = preds_index.transpose(1, 0)
                preds_index_reshape = preds_index.reshape(-1)
                preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

                for l in preds_sizes:
                    t = preds_index_reshape[index:index + l]

                    # NOTE: t might be zero size
                    if t.shape[0] == 0:
                       continue

                    char_list = []
                    for i in range(l):
                        # removing repeated characters and blank.
                        if not (i > 0 and t[i - 1] == t[i]):
                            if self.characters[t[i]] != '#':
                                char_list.append(self.characters[t[i]])
                    text = ''.join(char_list)
                    texts.append(text)

                    index += l

                return texts

       characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
       codec = CTCCodec(characters)
       ctrl = dai.CameraControl()
       ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
       ctrl.setAutoFocusTrigger()
       controlQueue.send(ctrl)

       while True:
            vid_in = q_vid.tryGet()
            if vid_in is not None:
                host_sync.add_msg(vid_in)

            # Multiple recognition results may be available, read until queue is empty
            while True:
                in_rec = q_rec.tryGet()
                if in_rec is None:
                   break
                rec_data = bboxes = np.array(in_rec.getFirstLayerFp16()).reshape(30,1,37)
                decoded_text = codec.decode(rec_data)[0]
                confidence=np.max(rec_data)
                pos = rotated_rectangles[rec_received]
                print("{:2}: {:20}".format(rec_received, decoded_text),
                    "center({:3},{:3}) size({:3},{:3}) angle{:5.1f} deg".format(
                        int(pos[0][0]), int(pos[0][1]), pos[1][0], pos[1][1], pos[2]))
            # Draw the text on the right side of 'cropped_stacked' - placeholder
                if cropped_stacked is not None:
                    cv2.putText(cropped_stacked, decoded_text,
                                (120 + 10 , 32 * rec_received + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow('cropped_stacked', cropped_stacked)
                rec_received += 1

                if confidence>0.8:
                    engine.say(decoded_text)
                    engine.runAndWait()

            if cv2.waitKey(1) == ord('q'):
                break

            if rec_received >= rec_pushed:
                in_det = q_det.tryGet()
                if in_det is not None:
                    frame = host_sync.get_msg(in_det.getSequenceNum()).getCvFrame().copy()

                    scores, geom1, geom2 = to_tensor_result(in_det).values()
                    scores = np.reshape(scores, (1, 1, 64, 64))
                    geom1 = np.reshape(geom1, (1, 4, 64, 64))
                    geom2 = np.reshape(geom2, (1, 1, 64, 64))

                    bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
                    boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
                    rotated_rectangles = [
                        east.get_cv_rotated_rect(bbox, angle * -1)
                        for (bbox, angle) in zip(boxes, angles)
                    ]

                    rec_received = 0
                    rec_pushed = len(rotated_rectangles)
                    if rec_pushed:
                        print("====== Pushing for recognition, count:", rec_pushed)
                    cropped_stacked = None
                    for idx, rotated_rect in enumerate(rotated_rectangles):
                        # Detections are done on 256x256 frames, we are sending back 1024x1024
                        # That's why we multiply center and size values by 4
                        rotated_rect[0][0] = rotated_rect[0][0] * 4
                        rotated_rect[0][1] = rotated_rect[0][1] * 4
                        rotated_rect[1][0] = rotated_rect[1][0] * 4
                        rotated_rect[1][1] = rotated_rect[1][1] * 4

                        # Draw detection crop area on input frame
                        points = np.int0(cv2.boxPoints(rotated_rect))
                        print(rotated_rect)
                        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

                        # TODO make it work taking args like in OpenCV:
                        # rr = ((256, 256), (128, 64), 30)
                        rr = dai.RotatedRect()
                        rr.center.x    = rotated_rect[0][0]
                        rr.center.y    = rotated_rect[0][1]
                        rr.size.width  = rotated_rect[1][0]
                        rr.size.height = rotated_rect[1][1]
                        rr.angle       = rotated_rect[2]
                        cfg = dai.ImageManipConfig()
                        cfg.setCropRotatedRect(rr, False)
                        cfg.setResize(120, 32)
                        # Send frame and config to device
                        if idx == 0:
                            w,h,c = frame.shape
                            imgFrame = dai.ImgFrame()
                            imgFrame.setData(to_planar(frame))
                            imgFrame.setType(dai.ImgFrame.Type.BGR888p)
                            imgFrame.setWidth(w)
                            imgFrame.setHeight(h)
                            q_manip_img.send(imgFrame)
                        else:
                            cfg.setReusePreviousImage(True)
                        q_manip_cfg.send(cfg)

                        # Get manipulated image from the device
                        transformed = q_manip_out.get().getCvFrame()

                        rec_placeholder_img = np.zeros((32, 200, 3), np.uint8)
                        transformed = np.hstack((transformed, rec_placeholder_img))
                        if cropped_stacked is None:
                            cropped_stacked = transformed
                        else:
                            cropped_stacked = np.vstack((cropped_stacked, transformed))
                    
                    

            if cropped_stacked is not None:
                cv2.imshow('cropped_stacked', cropped_stacked)

            if frame is not None:
                cv2.imshow('frame', frame)

            key = cv2.waitKey(1)
            if  key == ord('q'):
                break
            elif key == ord('t'):
               print("Autofocus trigger (and disable continuous)")
               ctrl = dai.CameraControl()
               ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
               ctrl.setAutoFocusTrigger()
               controlQueue.send(ctrl)
            if voice_command=="stop":        
               return

def read_news():
    # Use requests library to get the news website
    url = "https://www.bbc.com/news"
    response = requests.get(url)

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("h3", class_="gs-c-promo-heading__title")

    # Loop through the articles and read them one by one
    for i, article in enumerate(articles):
        title = article.text.strip()
        engine.say(f"News {i+1}: {title}")
        engine.runAndWait()

        # Ask for confirmation to read the next news article
        confirmation = get_confirmation()
        if confirmation == "no":
            break

    engine.say("That's all the news for now.")
    engine.runAndWait()

def get_confirmation():
    with sr.Microphone() as source:
        engine.say("Would you like to hear the next news? Say 'yes' or 'no'.")
        engine.runAndWait()
        audio = r.listen(source)

        try:
            confirmation = r.recognize_google(audio).lower()
            engine.say(f"You said: {confirmation}")
            engine.runAndWait()
            return confirmation
        except sr.UnknownValueError:
            engine.say("Could not understand your response.")
        except sr.RequestError as e:
            engine.say(f"Error occurred: {e}")

def get_weather(city):
    # OpenWeatherMap API key
    api_key = "ENTER_YOUR_API_KEY_HERE"

    # Request weather data for the given city
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        # Extract the temperature and weather description for today
        today_temp = int(data["main"]["temp"] - 273.15)
        today_weather = data["weather"][0]["description"]

        # Request weather forecast data for the given city
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            # Extract the temperature and weather description for tomorrow
            tomorrow_temp = int(data["list"][8]["main"]["temp"] - 273.15)
            tomorrow_weather = data["list"][8]["weather"][0]["description"]

            # Speak the weather information for the given city
            engine.say(f"Today's temperature in {city} is {today_temp} degrees Celsius with {today_weather}.")
            engine.say(f"Tomorrow's temperature in {city} is expected to be {tomorrow_temp} degrees Celsius with {tomorrow_weather}.")
            engine.runAndWait()
        else:
            engine.say("Failed to retrieve weather forecast. Please try again later.")
    else:
        engine.say("Failed to retrieve weather information. Please try again later.")

def voice_command():
    with sr.Microphone() as source:
        engine.say("What can I do for you?")
        engine.runAndWait()
        audio = r.listen(source)

        try:
            command = r.recognize_google(audio)
            engine.say(f"You said: {command}")
            engine.runAndWait()
            return command.lower()
        except sr.UnknownValueError:
            engine.say("Could not understand your command.")
        except sr.RequestError as e:
            engine.say(f"Error occurred: {e}")

def voice_output(text):
    engine.say(text)
    engine.runAndWait()

def wrapper_function():
    voice_output("Welcome to AIBA. I am made by Zunaira Khalid and Minahil Javed under the supervision of Dr. Yasir Awais.")
    

    while True:
        command = voice_command()

        if command == "detect objects":
            run_module("object detection")
        elif command == "tell weather":
            run_module("weather")
        elif command == "get news":
            run_module("news")
        elif command == "read":
            run_module("OCR")
        elif command == "exit":
            voice_output("Exiting...")
            break
        else:
            voice_output("Invalid command. Please try again.")

def run_module(module):
    if module == "object detection":
        # Object detection code here
        print("Running object detection module")
        run_object_detection()
    elif module == "weather":
        # Weather code here
        print("Running weather module")
        city = get_city()
        get_weather(city)
    elif module == "news":
        # News code here
        print("Running news module")
        read_news()
    elif module == "OCR":
        # News code here
        print("Running news module")
        OCR()
    
    else:
        print("Invalid module")

def get_city():
    with sr.Microphone() as source:
        engine.say("Sure! Which city's weather would you like to check?")
        engine.runAndWait()
        audio = r.listen(source)

        try:
            city = r.recognize_google(audio)
            engine.say(f"You said: {city}")
            engine.runAndWait()
            return city
        except sr.UnknownValueError:
            engine.say("Could not understand the city name.")
        except sr.RequestError as e:
            engine.say(f"Error occurred: {e}")

wrapper_function()