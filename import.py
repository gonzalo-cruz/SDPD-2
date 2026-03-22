import kagglehub

# Download latest version
path = kagglehub.dataset_download("stefanoleone992/tripadvisor-european-restaurants")

print("Path to dataset files:", path)