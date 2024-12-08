from PIL import Image

def process_image(input_image_path):
    # Open the image
    with Image.open(input_image_path) as img:
        # Resize the image to width: 96 and height: 160
        resized_img = img.resize((96, 160))
        
        # Save the resized image as a .bmp file
        resized_img.save("test.bmp", format="BMP")

if __name__ == "__main__":
    # Replace 'input_image.jpg' with the path to your input image
    input_image_path = 'ryan-reynolds-becomes-house-intruder-on-woman-in-gold-09.jpg'
    process_image(input_image_path)