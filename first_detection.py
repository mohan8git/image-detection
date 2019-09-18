from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "flower.jpg"), output_image_path=os.path.join(execution_path , "flower1.jpg"))
#here in above line replace the flower.jpg with image file you want to detect and then in place of flower1.jpg write the destination file name
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
