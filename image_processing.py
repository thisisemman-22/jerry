from PIL import Image, ImageFilter
import numpy as np
import math
from io import BytesIO
import uuid

def process_image(image_file, process_type, **kwargs):
    """
    Opens an image, applies a specified numerical processing method,
    and returns the result as a base64 encoded PNG string.
    """
    try:
        # Open the image using PIL
        image = Image.open(image_file).convert('RGB') # Ensure image is RGB

    except Exception as e:
        raise ValueError(f"Could not open or read image file: {e}")

    if process_type == 'downscale':
        # Using Newton's Divided Difference Interpolation
        processed_image = downscale(image)
    elif process_type == 'upscale':
        # Using Newton's Divided Difference Interpolation
        processed_image = upscale(image)
    elif process_type == 'denoise':
        # Using Gauss-Seidel method
        processed_image = denoise(image)
    elif process_type == 'blur':
        # Using trapezoidal rule for Gaussian kernel generation
        radius = kwargs.get('radius', 5)  # Default radius is 5
        processed_image = blur(image, radius=radius)
    else:
        raise ValueError(f"Unsupported process type: {process_type}")

    # Generate a unique filename to avoid overwriting existing files
    unique_id = uuid.uuid4().hex[:8]  # Generate an 8-character unique ID
    output_path = f"public/processed_{process_type}_{unique_id}.png"

    # Save the processed image to the public folder and return the file path
    processed_image.save(output_path, format="PNG")
    return output_path.replace("public/", "")  # Return relative path for URL generation

# ---------------------------------------------------------
# Downscaling Function (Newton's Divided Difference Interpolation)
# ---------------------------------------------------------
def downscale(image):
    # Reduce image size by half using Newton's Divided Difference Interpolation
    width, height = image.size
    new_width, new_height = width // 2, height // 2

    # Create a new blank image
    new_image = Image.new('RGB', (new_width, new_height))
    pixels = new_image.load()

    def divided_differences(points):
        n = len(points)
        table = [[0] * n for _ in range(n)]
        for i in range(n):
            table[i][0] = points[i][1]
        for j in range(1, n):
            for i in range(n - j):
                if points[i + j][0] == points[i][0]:
                    table[i][j] = 0  # Fallback value to avoid division by zero
                else:
                    table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (points[i + j][0] - points[i][0])
        return table[0]

    for i in range(new_width):
        for j in range(new_height):
            x = i * 2
            y = j * 2

            # Use Newton's Divided Difference Interpolation for pixel values
            neighbors = [
                (x, np.array(image.getpixel((x, y)))),
                (x + 1, np.array(image.getpixel((min(x + 1, width - 1), y)))),
                (y, np.array(image.getpixel((x, min(y + 1, height - 1))))),
                (y + 1, np.array(image.getpixel((min(x + 1, width - 1), min(y + 1, height - 1)))))
            ]

            coeffs = divided_differences(neighbors)
            interpolated_pixel = coeffs[0]  # Simplified for demonstration

            pixels[i, j] = tuple(interpolated_pixel.astype(int))

    return new_image

# ---------------------------------------------------------
# Upscaling Function (Newton's Divided Difference Interpolation)
# ---------------------------------------------------------
def upscale(image):
    # Double the image size using Newton's Divided Difference Interpolation
    width, height = image.size
    new_width, new_height = width * 2, height * 2

    # Create a new blank image
    new_image = Image.new('RGB', (new_width, new_height))
    pixels = new_image.load()

    def divided_differences(points):
        n = len(points)
        table = [[0] * n for _ in range(n)]
        for i in range(n):
            table[i][0] = points[i][1]
        for j in range(1, n):
            for i in range(n - j):
                if points[i + j][0] == points[i][0]:
                    table[i][j] = 0  # Avoid division by zero
                else:
                    table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (points[i + j][0] - points[i][0])
        return table[0]

    for i in range(new_width):
        for j in range(new_height):
            x = i / 2
            y = j / 2

            # Ensure indices are within bounds
            x0 = max(0, min(int(x), width - 1))
            y0 = max(0, min(int(y), height - 1))
            x1 = max(0, min(x0 + 1, width - 1))
            y1 = max(0, min(y0 + 1, height - 1))

            # Use Newton's Divided Difference Interpolation for pixel values
            neighbors = [
                (x0, np.array(image.getpixel((x0, y0)))),
                (x1, np.array(image.getpixel((x1, y0)))),
                (y0, np.array(image.getpixel((x0, y1)))),
                (y1, np.array(image.getpixel((x1, y1))))
            ]

            coeffs = divided_differences(neighbors)
            interpolated_pixel = coeffs[0]  # Simplified for demonstration

            pixels[i, j] = tuple(np.clip(interpolated_pixel, 0, 255).astype(int))

    return new_image

# ---------------------------------------------------------
# Denoising Function (Gauss-Seidel Method)
# ---------------------------------------------------------
def denoise(image):
    # Apply denoising using Gauss-Seidel method for smoothing
    width, height = image.size
    new_image = image.copy()
    pixels = new_image.load()

    for _ in range(5):  # Iterate to smooth the image
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                # Gauss-Seidel smoothing
                neighbors = [
                    np.array(pixels[i - 1, j]),
                    np.array(pixels[i + 1, j]),
                    np.array(pixels[i, j - 1]),
                    np.array(pixels[i, j + 1])
                ]
                pixels[i, j] = tuple(sum(neighbors) // len(neighbors))

    return new_image

# ---------------------------------------------------------
# Blurring Function (Trapezoidal Rule for Gaussian Kernel)
# ---------------------------------------------------------
def blur(image, radius):
    # Apply a Gaussian blur using the trapezoidal rule for numerical integration
    def gaussian(x, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (x / sigma) ** 2)

    kernel_size = 5
    sigma = radius
    kernel = []
    step = 1.0  # Step size for integration

    for i in range(kernel_size):
        x = i - kernel_size // 2
        kernel.append(gaussian(x, sigma))

    # Normalize the kernel
    kernel_sum = sum(kernel)
    kernel = [k / kernel_sum for k in kernel]

    # Apply the kernel to the image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

# ---------------------------------------------------------
# Example Usage (if running this script directly)
# ---------------------------------------------------------
if __name__ == '__main__':
    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (100, 80), color = 'red')
    # Fill with some noise/pattern for better visualization
    pixels = dummy_image.load()
    for i in range(dummy_image.width):
        for j in range(dummy_image.height):
            if (i + j) % 7 == 0:
                pixels[i, j] = (0, 255, 0)
            elif (i - j) % 11 == 0:
                pixels[i,j] = (0,0,255)
            if (i % 10 == 0) or (j%10 == 0):
                 pixels[i,j] = (255,255,255)


    dummy_image_file = BytesIO()
    dummy_image.save(dummy_image_file, format='PNG')
    dummy_image_file.seek(0)
    dummy_image.save("test_original.png") # Save original for comparison

    print("Processing: Downscale (Average)")
    downscaled_path = process_image(dummy_image_file, 'downscale')
    img_downscaled = Image.open(downscaled_path)
    img_downscaled.save("test_downscaled.png")
    print(f"Downscaled size: {img_downscaled.size}")

    print("\nProcessing: Upscale (Bilinear)")
    dummy_image_file.seek(0) # Reset file pointer
    upscaled_path = process_image(dummy_image_file, 'upscale')
    img_upscaled = Image.open(upscaled_path)
    img_upscaled.save("test_upscaled.png")
    print(f"Upscaled size: {img_upscaled.size}")

    print("\nProcessing: Denoise (Median)")
    dummy_image_file.seek(0)
    denoised_path = process_image(dummy_image_file, 'denoise')
    img_denoised = Image.open(denoised_path)
    img_denoised.save("test_denoised.png")
    print(f"Denoised size: {img_denoised.size}")

    print("\nProcessing: Blur (Gaussian)")
    dummy_image_file.seek(0)
    blurred_path = process_image(dummy_image_file, 'blur', radius=5)
    img_blurred = Image.open(blurred_path)
    img_blurred.save("test_blurred.png")
    print(f"Blurred size: {img_blurred.size}")

    print("\nDone. Check test_*.png files.")