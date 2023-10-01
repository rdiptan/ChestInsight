# import torch
# import torch.nn.functional as F
# from monai.visualize import GradCAM
# import matplotlib.pyplot as plt
# from src.full_model.generate_reports_for_images import get_model, get_image_tensor

# # Path to the checkpoint file
# checkpoint_path = "./full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt"

# # Path to the input image
# image_path = "./data/person88_virus_163_gray_scaled.jpg"

# # Load the model from the checkpoint
# model = get_model(checkpoint_path)

# # Set the model to evaluation mode
# model.eval()

# # Initialize the GradCAM object with the model's object_detector module and target layers
# cam = GradCAM(nn_module=model.object_detector, target_layers="backbone.7.2.conv3")

# # Get the image tensor from the image path
# image_tensor = get_image_tensor(image_path)

# # Enable gradient computation for the image tensor
# image_tensor.requires_grad_()

# # Perform inference on the image using the object_detector module
# output = model.object_detector.inference(image_tensor)

# # Add a singleton dimension to the output tensor
# output = torch.unsqueeze(output, dim=0)

# # Generate the heatmap using GradCAM
# heatmap = cam(output, class_idx=0)

# # Interpolate the heatmap to match the size of the input image
# heatmap = F.interpolate(heatmap, image_tensor.shape[2:], mode="bilinear", align_corners=False)

# # Convert the heatmap to a numpy array and squeeze the singleton dimensions
# heatmap = heatmap.squeeze().cpu().numpy()

# # Display the heatmap using the 'jet' colormap
# plt.imshow(heatmap, cmap='jet')
# plt.show()