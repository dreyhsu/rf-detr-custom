from PIL import Image
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
 
model = RFDETRMedium()
image = Image.open(r"D:\Admin\Downloads\COCO_train2014_000000003157.jpg")
detections = model.predict(image, threshold=0.5)
 
labels = [f"{COCO_CLASSES[class_id]} {conf:.2f}"
          for class_id, conf in zip(detections.class_id, detections.confidence)]
 
annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
 
annotated_image.save(r"D:\Dre\rf-detr\output.jpg")   # save to disk
annotated_image.show()       