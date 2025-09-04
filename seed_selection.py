import os
import torch
from torchvision import transforms
from PIL import Image
from cldm.get_intrinsic_seed import LatentIntrinsc

# Define the transformations to resize, convert to tensor, and normalize the image
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to the same height and width (H, W)
    transforms.ToTensor()       # Convert PIL Image to torch.Tensor (C, H, W)
])

def load_images_from_path(path):
    images = []
    files = []
    for filename in os.listdir(path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Add your file extensions
            img = Image.open(os.path.join(path, filename)).convert('RGB')  # Open and convert to RGB
            img_tensor = transform(img)  # Apply transformations (C, H, W)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)
            images.append(img_tensor)
            files.append(filename.strip().split('i')[0])
    # Stack all images to form a tensor of shape [B, C, H, W]
    return torch.cat(images, dim=0), files

# Load images from Path A
path_A = PATH_TO_RELIT_IMAGE_FLODER
# H, W = 256, 256  # Height and width of the images
images_A,files = load_images_from_path(path_A)
B,C,H,W = images_A.shape
# Load the single image from Path B
path_B = PATH_TO_REFERENCE_IMAGE
single_image_B = Image.open(path_B).convert('RGB')
single_image_B_tensor = transform(single_image_B)  # Convert to tensor (C, H, W)
single_image_B_tensor = single_image_B_tensor.unsqueeze(0).repeat(B, 1, 1, 1)  # Repeat to match batch size [B, C, H, W]

# Concatenate the tensors channel-wise (i.e., along dimension 1)
combined_tensor = torch.cat((images_A, single_image_B_tensor), dim=1)

print(f"Combined Tensor Shape: {combined_tensor.shape}")  # Should print [B, 6, H, W] if both tensors had 3 channels
combined_tensor = combined_tensor.cuda()*2-1
model = LatentIntrinsc()
model = model.cuda().eval()
diff_list = []
for i in range(B):
    input_img = combined_tensor[i,...]
    input_img = input_img.unsqueeze(0)
    diff = model(input_img)
    diff_list.append(diff.item())


data_dict = {'Diffs': diff_list, 'seed': files}

# Zip A and B together and sort them based on values of A
sorted_pairs = sorted(zip(data_dict['Diffs'], data_dict['seed']), key=lambda x: x[0])

# Unzip the sorted pairs back into two lists
sorted_A, sorted_B = zip(*sorted_pairs)
top_1_A = sorted_A[1]
top_1_B = sorted_B[1]

top_2_A = sorted_A[2]
top_2_B = sorted_B[2]

top_3_A = sorted_A[3]
top_3_B = sorted_B[3]

top_4_A = sorted_A[4]
top_4_B = sorted_B[4]

top_5_A = sorted_A[5]
top_5_B = sorted_B[5]


# Update the dictionary with the top-5 values
data_dict['Diffs'] = list([top_1_A,top_2_A,top_3_A,top_4_A,top_5_A])
data_dict['seed'] = list([top_1_B,top_2_B,top_3_B,top_4_B,top_5_B])

print("Top-5 A:", data_dict['Diffs'])
print("Corresponding B:", data_dict['seed'])
