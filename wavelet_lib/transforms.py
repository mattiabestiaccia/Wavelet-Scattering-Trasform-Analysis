import torch
from torchvision import transforms
from PIL import Image

def process_image_for_classification(image_path: str) -> torch.Tensor:
    """
    Process an image for classification with the wavelet model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        torch.Tensor: Processed image tensor ready for model input
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image)