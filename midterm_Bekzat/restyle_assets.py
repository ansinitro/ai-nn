import os
import glob
from PIL import Image, ImageEnhance

def color_shift_image(image_path):
    try:
        # Open the image
        img = Image.open(image_path).convert('RGB')
        
        # Split into R, G, B channels
        r, g, b = img.split()
        
        # We can simulate a style change by swapping channels or blending
        # Let's do a channel swap: G->R, B->G, R->B for colored pixels, 
        # but wait, backgrounds are usually white. White is (255, 255, 255), so swap doesn't affect it!
        # This safely changes plot line colors while keeping black/white intact.
        shifted_img = Image.merge('RGB', (g, b, r))
        
        # Increase contrast slightly to make it look "premium"
        enhancer = ImageEnhance.Contrast(shifted_img)
        shifted_img = enhancer.enhance(1.2)
        
        # Save back
        shifted_img.save(image_path)
        print(f"Shifted colors for {image_path}")
    except Exception as e:
        print(f"Skipping {image_path}: {e}")

assets_dir = "/home/titan/Documents/master/ai-nn/midterm_Bekzat/Labs_notebooks/assets"
images = glob.glob(os.path.join(assets_dir, "*.png"))

for img_path in images:
    color_shift_image(img_path)

print("Asset restyling complete!")
