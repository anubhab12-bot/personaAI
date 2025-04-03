import cloudinary
import cloudinary.api
import cloudinary.uploader

cloudinary.config(
    cloud_name="darid8ehu",
    api_key="343657836156578",
    api_secret="qhgy8NrZuaY8ZxW3Lwhkl_SgyWo",
    secure=True,
)

def upload_image(image_path):
    """Upload an image to Cloudinary and return the URL."""
    try:
        response = cloudinary.uploader.upload(image_path)
        image_url = response['secure_url']  # Use 'secure_url' for HTTPS
        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None

# image_path = r"C:/Users/anubh/Downloads/Happy Holiday (1).png" 

# uploaded_image_url = upload_image(image_path)

# if uploaded_image_url:
#     print(f"Image uploaded successfully! URL: {uploaded_image_url}")
# else:
#     print("Image upload failed.")

