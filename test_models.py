from models import (
    get_resnet50,
    get_efficientnet,
    get_vit
)


models = [
    get_resnet50(),
    get_efficientnet(),
    get_vit()
]


for model in models:
    print(type(model).__name__)
    print("Loaded successfully ✅")