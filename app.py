%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Chẩn đoán Bệnh Da Liễu", layout="centered")
st.title("🩺 Chẩn đoán Bệnh Da Liễu")
st.markdown("**Mô hình: EfficientNet-B2 + CBAM** - Đồ án tốt nghiệp")

# ====================== CBAM & Model ======================
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([avg_out, max_out], dim=1))

class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        from torchvision.models import efficientnet_b2
        base = efficientnet_b2(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(nn.Dropout(0.68), nn.Linear(1408, num_classes))
        self.cbams = nn.ModuleList([CBAM(88), CBAM(120), CBAM(208), CBAM(352)])

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if 4 <= i <= 7:
                x = self.cbams[i-4](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Load model từ Google Drive
@st.cache_resource
def load_model():
    model_path = "efficientnet_b2_cbam_best.pth"
    if not os.path.exists(model_path):
        st.info("🔄 Đang tải mô hình từ Google Drive (lần đầu có thể mất 30-60 giây)...")
        url = "https://drive.google.com/uc?id=1nvzGwzw4rvlI8Oqontw01e9zcOquN_Wv"
        gdown.download(url, model_path, quiet=False)
    
    model = EfficientNetCBAM(num_classes=23)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model

model = load_model()

# Class names
class_names = ["Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma", "Atopic Dermatitis Photos", 
               "Bullous Disease Photos", "Cellulitis Impetigo", "Eczema Photos", "Exanthems and Drug Eruptions", 
               "Hair Loss Photos", "Herpes HPV", "Light Diseases", "Lupus", "Melanoma", "Nail Fungus", 
               "Poison Ivy", "Psoriasis Lichen Planus", "Rosacea", "Seborrheic Keratoses", "Systemic Disease", 
               "Tinea Ringworm", "Urticaria Hives", "Vascular Tumors", "Vasculitis", "Warts Molluscum"]

# Transform
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("📤 Upload ảnh da cần chẩn đoán", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probs, 3)

    st.subheader("🔍 Kết quả dự đoán")
    for i in range(3):
        st.metric(label=f"Top {i+1}: **{class_names[top_idx[i]]}**", value=f"{top_prob[i].item()*100:.2f}%")

else:
    st.info("👆 Vui lòng upload ảnh da")

st.caption("Ứng dụng hỗ trợ chẩn đoán bệnh da liễu - Đồ án tốt nghiệp")