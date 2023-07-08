from model import Discriminator
import torch
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import argparse
from tqdm import tqdm
import csv
import datetime

transform_input = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def merge_images(img1, img2):
   # Resize img and c_map to have the same height
    height = max(img1.shape[0], img2.shape[0])
    resized_img = cv2.resize(img1, (int(int(img1.shape[1]) * height / int(img1.shape[0])), height), interpolation=cv2.INTER_LINEAR)
    resized_c_map = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height), interpolation=cv2.INTER_NEAREST)
    width = resized_img.shape[1] + resized_c_map.shape[1]
    merged_img = np.zeros((height, width, 3), dtype=np.uint8)
    merged_img[:, :resized_img.shape[1]] = resized_img
    merged_img[:, resized_img.shape[1]:] = resized_c_map

    return merged_img

def get_model():
    discriminator = Discriminator().cuda()
    model_file_path = "./weights/IFQA_Metric.pth"
    discriminator.load_state_dict(torch.load(model_file_path)['D'])
    discriminator.eval()
    return discriminator

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_input(image=img)['image'].unsqueeze(0).cuda()
    return img

def evaluate_img(img_path, model, save_data, timestamp):
    img = read_image(img_path)
    with torch.no_grad():
        p_map = model(img)
        sum = torch.sum(p_map) / (256*256)
        score = round(torch.mean(sum).detach().cpu().item(), 4)
    p_map = p_map.squeeze().cpu().numpy()
    p_map = p_map[..., np.newaxis]
    p_map *= 255
    p_map = p_map.astype(np.uint8)
    c_map = cv2.applyColorMap(p_map, colormap=16)
    saved_name = os.path.splitext(os.path.basename(img_path))[0]
    if (save_data):
        img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = np.clip((img_np * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_map = merge_images(img_np, c_map)
        result_folder = f"./results/{timestamp}/"
        os.makedirs(result_folder, exist_ok=True)
        cv2.imwrite("{0}{1}.png".format(result_folder, saved_name), img_map)
    else:
        print("File: {0} Score: {1}".format(saved_name, score))
    return saved_name, score

def main(config):
    discriminator = get_model()
    f_path = config.f_path
    csv_path = config.csv_path
    save_data = config.save_data
    images_score = []
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    print("Start assessment..")
    if os.path.isfile(f_path):
        saved_name, score = evaluate_img(f_path, discriminator, save_data, timestamp)
        images_score.append((saved_name, score))
    else:
        files_list = os.walk(f_path).__next__()[2]
        for file_name in tqdm(files_list):
            file_path = os.path.join(f_path, file_name)
            saved_name, score = evaluate_img(file_path, discriminator, save_data, timestamp)
            images_score.append((saved_name, score))

    print(f"Saving at {csv_path}")
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        scores = [score for _, score in images_score]  # Extract scores into a separate list

        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        writer.writerow(["Statistics"])
        writer.writerow(["Mean", mean_score])
        writer.writerow(["Standard Deviation", std_dev])

        writer.writerow(["Image", "Score"])
        for name, score in images_score:
            writer.writerow([name, score])
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, default="./docs/toy", help='file path or folder path')
    parser.add_argument('--csv_path', type=str, default="./results/result.csv", help='csv results file name')
    parser.add_argument('--save_data', type=str, default=False, help='boolean to save images and csv')
    config = parser.parse_args()
    print(config)
    main(config)
