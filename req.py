import requests

# Specify the path to the image file you want to test
image_path = r'C:\Users\Anubhav\Desktop\Major_Project_2\venv\WhatsApp Image 2023-12-04 at 9.16.58 PM.jpeg'

# Create a dictionary with the file to be uploaded
files = {'image': open(image_path, 'rb')}  # Use 'image' as the key

# Make a POST request to the /upload_image endpoint
response = requests.post('http://127.0.0.1:5000/upload_image', files=files)

# Print the response
print(response.text)
