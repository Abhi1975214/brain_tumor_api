# # utils/predict.py
# import torch
# from torchvision import transforms,models
# from PIL import Image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.efficientnet_b2(weights=None)  # weights=None to avoid pretrained weights

# # Adjust the classifier for 4 output classes
# num_features = model.classifier[1].in_features
# model.classifier[1] = torch.nn.Linear(num_features,3)

# # Load your trained weights
# model.load_state_dict(torch.load("model/model.pth", map_location=device))
# model.to(device)
# model.eval()
# def predict_image(img: Image.Image):    # Open image
#     image_transform = transforms.Compose(
#         [
#             transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#             ),
#         ]
#     )
#     model.eval()
#     with torch.inference_mode():
#         transformed_image = image_transform(img).unsqueeze(dim=0)
#         target_image_pred = model(transformed_image.to(device))
#     target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
#     target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1).item()
#     return target_image_pred_label
# predict.py
import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B2 model
model = models.efficientnet_b2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 4)  # 3 output classes

# Load the trained weights
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.to(device)
model.eval()

# Label map
label_map = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

def predict_image(img: Image.Image):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        output = model(transformed_image.to(device))
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        # predicted_label =torch.argmax(probabilities, dim=1).item()
        predicted_label = int(probabilities.argmax())
    return predicted_label, probabilities
