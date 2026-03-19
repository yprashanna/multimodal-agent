import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection

class ImageProcessor:
    def __init__(self):
        # Initialize BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Initialize DETR for object detection
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
    def get_caption(self, image: Image.Image) -> str:
        """Generate a natural language caption for the image"""
        inputs = self.blip_processor(image, return_tensors="pt")
        out = self.blip_model.generate(**inputs, max_length=50)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)
    
    def detect_objects(self, image: Image.Image) -> str:
        """Detect objects in the image and return formatted results"""
        inputs = self.detr_processor(images=image, return_tensors="pt")
        outputs = self.detr_model(**inputs)
        
        # Post-process to get readable results
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "object": self.detr_model.config.id2label[label.item()],
                "confidence": f"{score.item():.2f}",
                "location": [int(x) for x in box.tolist()]
            })
        return detections