import torch
import clip
from PIL import Image
import numpy as np

def class_demo():
    # 测试分类的demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("ViT-B/32", device=device)  # 载入模型
    image = preprocess(Image.open("./cat_dog.jpeg")).unsqueeze(0).to(device)
    text_language = ["a diagram", "a dog", "a black cat", "two dogs", "two cats", "dog and cat"]
    text = clip.tokenize(text_language).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        idx = np.argmax(probs, axis=1)
        for i in range(image.shape[0]):
            id = idx[i]
            print('image {}\tlabel\t{}:\t{}'.format(i, text_language[id],probs[i,id]))
            print('image {}:\t{}'.format(i, [v for v in zip(text_language,probs[i])]))


if __name__ == '__main__':
    class_demo()
