import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', type=str)
parser.add_argument('-c', '--category', type=str, default='material', help="Category to extract the binary mask for")
args = parser.parse_args()


def load_dataset_info(dataset_path):
    """
    Load dataset information from dataset path including the classes and their related labels
    """
    dataset_path_elements = os.listdir(dataset_path)

    class_label_file = glob.glob(os.path.join(dataset_path, "List*.csv"))[0]
    class_label_path = os.path.join(dataset_path, str(class_label_file))
    if class_label_path is None:
        raise Exception("CSV file describing the classes not found in the following path: {}".format(dataset_path))
    
    class_label_df = pd.read_csv(class_label_path)
    class_label_dict = dict(zip(class_label_df['Desc'], 
                            class_label_df['Label']))
    return dataset_path_elements, class_label_df, class_label_dict

def visualize_mask_overlay(image, mask, category):
    """
    Overlay the binary mask on the image with transparency.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Left subplot: Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original EL images")
    axes[0].axis("off")

    # Right subplot: Image with overlayed mask
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap='Reds', alpha=0.5)
    axes[1].set_title(f"Mask: {category}")
    axes[1].axis("off")
    
    # Right subplot: Image with overlayed mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Binary masked: {category}")
    axes[2].axis("off")


    plt.tight_layout()
    plt.show()

def extract_binary_mask(dataset_path, category, visualize=True, save_masks=False):
    """
    Extract the binary masks and visualize them"""
    dataset_path_elements, class_label_df, class_label_dict = load_dataset_info(dataset_path)
    splits = ['train', 'val', 'test']
    for split in splits:
        masks_folder = [element for element in dataset_path_elements if 'masks' in element and split in element][0]
        images_folder = [element for element in dataset_path_elements if 'images' in element and split in element][0]
        
        if len(masks_folder) == 0:
            raise ValueError("Mask folder for training not found in the following path: {}".format(self.dataset_path))
        
        for mask in masks_folder:
            
            masks_path = os.path.join(dataset_path, masks_folder)
            masks = [f for f in os.listdir(masks_path) if f.endswith(('.jpg','.png','.jpeg'))]
        
            images_path = os.path.join(dataset_path, images_folder)
            images = [f for f in os.listdir(images_path) if f.endswith(('.jpg','.png','.jpeg'))]
                
            for image in images:
                if 'rotate' not in image and 'mirror' not in image and 'flip' not in image:
                    mask = [mask for mask in masks if mask.split('.')[0] == image.split('.')[0]][0]
                    image_path = os.path.join(images_path, image)
                    mask_path = os.path.join(masks_path, mask)
                    
                    image_img = cv2.imread(image_path)  
                    image_img = cv2.cvtColor(image_img, cv2.COLOR_BGR2RGB)
                    
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    label = class_label_dict[category]
                    
                    if label in mask_img:
                        binary_mask = np.where(mask_img == label, 255, 0).astype(np.uint8)
                        binary_mask = cv2.resize(binary_mask, (image_img.shape[0], image_img.shape[1]))
                        
                        if save_masks:
                            binary_mask_path = os.path.join(masks_path, f'binary_masks_{category}', f'{image.split(".")[0]}_{category}.png')
                            os.makedirs(os.path.dirname(binary_mask_path), exist_ok=True)
                            cv2.imwrite(binary_mask_path, binary_mask) 
                        
                        # Visualization
                        if visualize:
                            visualize_mask_overlay(image_img, binary_mask, category)    
                        
extract_binary_mask(args.dataset_path, args.category, visualize=True, save_masks=False)                   
