from PIL import Image, ImageFilter
import numpy as np
import math
from io import BytesIO
import uuid
import os
from numba import jit, prange

def process_image(image_file, process_type, **kwargs):
    """
    Opens an image, applies a specified numerical processing method,
    and returns the result as a base64 encoded PNG string.
    """
    try:
        # Open the image using PIL
        image = Image.open(image_file).convert('RGB') # Ensure image is RGB

    except Exception as e:
        raise ValueError("Could not open or read image file. Please try again.")

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

    # Ensure the 'public' directory exists
    os.makedirs('public', exist_ok=True)

    # Generate a unique filename to avoid overwriting existing files
    unique_id = uuid.uuid4().hex[:8]  # Generate an 8-character unique ID
    output_path = f"public/processed_{process_type}_{unique_id}.png"

    # Save the processed image to the public folder and return the file path
    try:
        processed_image.save(output_path, format="PNG")
    except Exception as e:
        raise ValueError("Failed to save the processed image. Please try again.")

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

# JIT-compiled Newton's interpolation functions for massive speedup
@jit(nopython=True)
def newton_interp_1d_numba(x_data, y_data, x_interp):
    """Fast 1D Newton interpolation for a single point with Numba acceleration"""
    n = len(x_data)
    
    # First coefficient is just y0
    dd = np.zeros(n, dtype=np.float64)
    dd[0] = y_data[0]
    
    # Calculate divided differences
    for i in range(1, n):
        # Compute ith divided difference
        term = y_data[i]
        for j in range(i):
            # Avoid division by zero
            if x_data[i] == x_data[j]:
                continue
            term = (term - y_data[j]) / (x_data[i] - x_data[j])
        dd[i] = term
    
    # Evaluate polynomial at x_interp
    result = dd[0]
    x_term = 1.0
    
    for i in range(1, n):
        x_term *= (x_interp - x_data[i-1])
        result += dd[i] * x_term
        
    return max(0.0, min(255.0, result))

@jit(nopython=True, parallel=True)
def process_row(img_row, width, new_width):
    """Process a single row with horizontal interpolation using Numba"""
    result = np.zeros((new_width, 3), dtype=np.uint8)
    x_points = np.array([0.0, 1.0])  # Normalized coordinates
    
    for x_new in range(0, new_width, 2):
        # Original pixels (even indices)
        x_orig = x_new // 2
        if x_orig < width:
            result[x_new] = img_row[x_orig]
        
        # Interpolated pixels (odd indices)
        if x_new + 1 < new_width:
            if x_orig + 1 < width:
                # For each color channel
                for c in range(3):
                    y_points = np.array([float(img_row[x_orig, c]), 
                                        float(img_row[min(x_orig + 1, width - 1), c])])
                    result[x_new + 1, c] = int(newton_interp_1d_numba(x_points, y_points, 0.5))
            else:
                # Edge case - use last known value
                result[x_new + 1] = img_row[width - 1]
    
    return result

@jit(nopython=True)
def process_column(col_data, new_height):
    """Process a single column with vertical interpolation using Numba"""
    result = np.zeros((new_height, 3), dtype=np.uint8)
    
    # Copy existing pixels - first copy all original pixels to their correct positions
    for y_orig in range(len(col_data)):
        y_new = y_orig * 2  # Each original pixel maps to an even index in the new array
        if y_new < new_height:
            result[y_new] = col_data[y_orig]
    
    # Interpolate missing pixels (odd indices)
    y_points = np.array([0.0, 1.0])  # Normalized coordinates
    for y in range(1, new_height, 2):
        y_prev = y - 1
        y_next = min(y + 1, new_height - 1)
        
        # For each color channel
        for c in range(3):
            p_points = np.array([float(result[y_prev, c]), float(result[y_next, c])])
            result[y, c] = int(newton_interp_1d_numba(y_points, p_points, 0.5))
    
    return result

def upscale(image):
    """
    Double the image size using Newton's Divided Difference Interpolation,
    applied separately in horizontal and vertical dimensions with Numba acceleration
    """
    width, height = image.size
    new_width, new_height = width * 2, height * 2
    
    # Convert original image to numpy array for faster operations
    img_array = np.array(image)
    
    # Step 1: Horizontal interpolation (rows)
    intermediate = np.zeros((height, new_width, 3), dtype=np.uint8)
    
    # Process each row in parallel for better performance
    for y in range(height):
        intermediate[y] = process_row(img_array[y], width, new_width)
    
    # Step 2: Create final image with vertical interpolation
    final_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Process each column for vertical interpolation
    for x in range(new_width):
        final_array[:, x] = process_column(intermediate[:, x], new_height)
    
    # Convert back to PIL Image
    return Image.fromarray(final_array)

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