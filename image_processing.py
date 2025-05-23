from PIL import Image, ImageFilter
import numpy as np
import math
from io import BytesIO
import uuid
import os
import time
from numba import jit, prange, set_num_threads
import multiprocessing

# Set Numba to use multiple threads for parallel operations
cpu_count = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
set_num_threads(cpu_count)

def process_image(image_file, process_type, **kwargs):
    """
    Opens an image, applies a specified numerical processing method,
    and returns the result as a base64 encoded PNG string.
    
    Additional parameters can be passed via kwargs:
    - edge_threshold: Edge sensitivity threshold for denoise function (default: 30)
                     Higher values preserve more detail
    - iterations: Number of iterations for denoise function (default: 1)
    - radius: Blur radius (default: 5)
    - sharpness: Detail enhancement for upscale and downscale (0.0-1.0, default: 0.6)
    """
    try:
        # Open the image using PIL
        image = Image.open(image_file).convert('RGB') # Ensure image is RGB
        print(f"Processing {process_type} on image of size {image.size} ({image.size[0]*image.size[1]} pixels)")
        
        # Start timer for performance measurement
        start_time = time.time()

    except Exception as e:
        raise ValueError("Could not open or read image file. Please try again.")

    if process_type == 'downscale':
        # Using Newton's Divided Difference Interpolation
        processed_image = downscale(image)
    elif process_type == 'upscale':
        # Using Newton's Divided Difference Interpolation
        processed_image = upscale(image)
    elif process_type == 'denoise':
        # Using Bilateral Gaussian Filter with configurable parameters
        edge_threshold = kwargs.get('edge_threshold', 30)
        iterations = kwargs.get('iterations', 1)  # Default to 1 iteration for Gaussian
        
        # Start iteration-specific timer for denoise operation
        denoise_start = time.time()
        processed_image = denoise(image, edge_threshold=edge_threshold, iterations=iterations)
        denoise_time = time.time() - denoise_start
        print(f"Denoise completed in {denoise_time:.2f} seconds using {cpu_count} threads")
    
    elif process_type == 'blur':
        # Using trapezoidal rule for Gaussian kernel generation
        radius = kwargs.get('radius', 5)  # Default radius is 5
        processed_image = blur(image, radius=radius)
    else:
        raise ValueError(f"Unsupported process type: {process_type}")
        
    # Total processing time
    total_time = time.time() - start_time
    print(f"Total {process_type} processing time: {total_time:.2f} seconds")

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
@jit(nopython=True)
def compute_2x2_newton_interpolation_fast(block):
    """
    Optimized Newton's Divided Difference Interpolation for a 2x2 block.
    Uses faster array operations while maintaining numerical accuracy.
    """
    # Pre-allocate result array
    result = np.zeros(3, dtype=np.float64)
    edge_threshold = 30.0
    
    # Process each channel with optimized operations
    for c in range(3):
        # Extract values as 2x2 array for current channel
        values = block[:, :, c].astype(np.float64)
        f00, f01 = values[0, 0], values[0, 1]
        f10, f11 = values[1, 0], values[1, 1]
        
        # Fast edge detection using min/max
        min_val = min(f00, f01, f10, f11)
        max_val = max(f00, f01, f10, f11)
        max_diff = max_val - min_val
        
        if max_diff > edge_threshold:
            # Quick average calculation
            avg = (f00 + f01 + f10 + f11) * 0.25
            
            # Compute absolute differences using Numba-compatible operations
            diff_values = np.zeros(4, dtype=np.float64)
            diff_values[0] = abs(f00 - avg)
            diff_values[1] = abs(f01 - avg)
            diff_values[2] = abs(f10 - avg)
            diff_values[3] = abs(f11 - avg)
            
            # Find maximum difference index using loop
            max_idx = 0
            max_diff_val = diff_values[0]
            for i in range(1, 4):
                if diff_values[i] > max_diff_val:
                    max_diff_val = diff_values[i]
                    max_idx = i
            
            # Apply weights directly
            weights = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float64)
            weights[max_idx] = 0.55
            
            # Weighted sum
            result[c] = (f00 * weights[0] + f01 * weights[1] + 
                        f10 * weights[2] + f11 * weights[3])
        else:
            # Optimized Newton interpolation for smooth areas
            # Combined divided differences calculation
            f10_00 = f10 - f00
            f01_00 = f01 - f00
            result[c] = f00 + 0.5 * (f10_00 + f01_00) + 0.25 * ((f11 - f10) - (f01 - f00))
        
        # Clip values
        result[c] = max(0.0, min(255.0, result[c]))
    
    return result.astype(np.uint8)

@jit(nopython=True, parallel=True)
def process_downscale_fast(img_array, new_width, new_height):
    """Optimized parallel processing for downscaling"""
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Process rows in parallel with better memory access pattern
    for j in prange(new_height):
        y_start = j * 2
        for i in range(new_width):
            x_start = i * 2
            
            # Check boundary condition once
            if y_start + 1 >= img_array.shape[0] or x_start + 1 >= img_array.shape[1]:
                result[j, i] = img_array[min(y_start, img_array.shape[0]-1), 
                                       min(x_start, img_array.shape[1]-1)]
                continue
            
            # Extract block with direct indexing
            block = img_array[y_start:y_start+2, x_start:x_start+2]
            result[j, i] = compute_2x2_newton_interpolation_fast(block)
    
    return result

def downscale(image):
    """
    Optimized downscaling using Newton's Divided Difference Interpolation.
    """
    width, height = image.size
    new_width, new_height = width // 2, height // 2
    
    # Convert to NumPy array with optimal memory layout
    img_array = np.ascontiguousarray(np.array(image))
    
    # Process using optimized functions
    result = process_downscale_fast(img_array, new_width, new_height)
    
    return Image.fromarray(result)

# ---------------------------------------------------------
# Upscaling Function (Newton's Divided Difference Interpolation)
# ---------------------------------------------------------

# JIT-compiled Newton's interpolation functions for massive speedup
@jit(nopython=True)
def newton_interp_1d_numba(x_data, y_data, x_interp):
    """
    Fast 1D Newton interpolation for a single point with Numba acceleration.
    Enhanced to preserve more details by favoring sharper transitions.
    """
    n = len(x_data)
    
    # For simple linear interpolation of 2 points, use a direct formula with sharpness enhancement
    if n == 2:
        # Calculate distance as 0 to 1
        t = (x_interp - x_data[0]) / (x_data[1] - x_data[0]) if x_data[1] != x_data[0] else 0.5
        
        # Apply slight non-linear transformation to t to enhance sharpness
        # This makes transitions between pixels more defined rather than blurred
        if t < 0.5:
            t = 0.4 + (t * 1.2)  # Sharper transition near original pixels
        
        # Linear interpolation with enhanced transitions
        return y_data[0] * (1.0 - t) + y_data[1] * t
    
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
# Blurring Function (Trapezoidal Rule for Gaussian Kernel)
# ---------------------------------------------------------
@jit(nopython=True)
def gaussian(x, sigma):
    """Compute Gaussian function with Numba acceleration"""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (x / sigma) ** 2)

@jit(nopython=True)
def create_gaussian_kernel_1d(radius, sigma):
    """Create a 1D Gaussian kernel using trapezoidal rule with Numba acceleration"""
    kernel_size = 2 * radius + 1
    kernel = np.zeros(kernel_size, dtype=np.float64)
    
    # Apply trapezoidal rule for numerical integration
    dx = 1.0  # Step size for integration
    for i in range(kernel_size):
        x = i - radius
        # Use trapezoidal rule: (f(a) + f(b)) * h / 2
        left = gaussian(x - dx/2, sigma)
        right = gaussian(x + dx/2, sigma)
        kernel[i] = (left + right) * dx / 2
    
    # Normalize the kernel to ensure it sums to 1
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    
    return kernel

@jit(nopython=True)
def apply_blur_1d(src, kernel, direction):
    """Apply 1D Gaussian blur in specified direction using the computed kernel"""
    height, width, channels = src.shape
    radius = len(kernel) // 2
    result = np.zeros_like(src, dtype=np.float64)
    
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                value = 0.0
                if direction == 0:  # Horizontal
                    for k in range(-radius, radius + 1):
                        x_k = max(0, min(width - 1, x + k))
                        value += src[y, x_k, c] * kernel[k + radius]
                else:  # Vertical
                    for k in range(-radius, radius + 1):
                        y_k = max(0, min(height - 1, y + k))
                        value += src[y_k, x, c] * kernel[k + radius]
                result[y, x, c] = value
                
    return result

def blur(image, radius=5):
    """
    Apply Gaussian blur using separable 1D convolutions.
    The kernel is computed using the trapezoidal rule for numerical integration.
    
    Parameters:
    - image: Input PIL image
    - radius: Blur radius (default: 5)
    """
    # Convert to NumPy array for processing
    img_array = np.array(image).astype(np.float64)
    
    # For small radius, use standard Gaussian calculation
    if radius <= 2:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # Calculate sigma based on radius (standard practice)
    sigma = radius / 2.0
    
    # Create 1D Gaussian kernel using trapezoidal rule
    kernel = create_gaussian_kernel_1d(radius, sigma)
    
    # Apply separable convolution
    # First horizontal pass
    temp = apply_blur_1d(img_array, kernel, 0)
    # Then vertical pass
    result = apply_blur_1d(temp, kernel, 1)
    
    # Clip values and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)

# ---------------------------------------------------------
# Denoising Function (Bilateral Gaussian Filter - Optimized for Speed)
# ---------------------------------------------------------

@jit(nopython=True)
def fast_spatial_weights(radius, sigma):
    """Precompute spatial weights for the bilateral filter"""
    size = 2 * radius + 1
    weights = np.zeros((size, size), dtype=np.float32)
    sigma_sq = sigma * sigma
    
    for i in range(size):
        for j in range(size):
            x = i - radius
            y = j - radius
            dist_sq = x*x + y*y
            weights[i, j] = np.exp(-0.5 * dist_sq / sigma_sq)
            
    return weights

@jit(nopython=True, parallel=True)
def process_tile(tile, spatial_weights, color_sigma, radius):
    """Process a single tile with the bilateral filter"""
    h, w, c = tile.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    
    # Add epsilon for numerical stability
    eps = 1e-8
    
    # Process each pixel in the tile (excluding borders)
    for y in prange(radius, h-radius):
        for x in range(radius, w-radius):
            # Process each color channel
            for channel in range(c):
                center = float(tile[y, x, channel])
                weighted_sum = 0.0
                weight_sum = 0.0
                
                # Use a smaller, fixed window for very large images
                for wy in range(-radius, radius+1):
                    for wx in range(-radius, radius+1):
                        # Get neighbor pixel
                        ny, nx = y + wy, x + wx
                        neighbor = float(tile[ny, nx, channel])
                        
                        # Spatial weight (precomputed)
                        s_weight = spatial_weights[wy+radius, wx+radius]
                        
                        # Range/color weight - simplified calculation
                        diff = center - neighbor
                        r_weight = np.exp(-0.5 * diff * diff / (color_sigma * color_sigma + eps))
                        
                        # Combined weight
                        weight = s_weight * r_weight
                        
                        weighted_sum += neighbor * weight
                        weight_sum += weight
                
                # Set output value with numerical stability check
                if weight_sum > eps:
                    result[y, x, channel] = np.uint8(weighted_sum / weight_sum)
                else:
                    result[y, x, channel] = tile[y, x, channel]
    
    # Copy border pixels (unprocessed) from original
    # Top and bottom borders
    for y in range(radius):
        for x in range(w):
            for channel in range(c):
                # Top border
                result[y, x, channel] = tile[y, x, channel]
                # Bottom border
                result[h-y-1, x, channel] = tile[h-y-1, x, channel]
    
    # Left and right borders (but not corners which were already copied)
    for y in range(radius, h-radius):
        for x in range(radius):
            for channel in range(c):
                # Left border
                result[y, x, channel] = tile[y, x, channel]
                # Right border
                result[y, w-x-1, channel] = tile[y, w-x-1, channel]
                
    return result
    
    return result

@jit(nopython=True)
def fast_bilateral_filter(img, spatial_sigma, color_sigma, use_smaller_window=True):
    """Optimized bilateral filter implementation"""
    height, width, channels = img.shape
    
    # Adjust window size based on image dimensions
    if use_smaller_window and (height > 1000 or width > 1000):
        # Use smaller window for large images
        spatial_sigma = 1.5
        radius = 2  # Smaller radius for large images = much faster
    else:
        radius = max(1, int(spatial_sigma * 1.5))  # Reduced from 2.0 to 1.5
    
    # Precompute spatial weights
    spatial_weights = fast_spatial_weights(radius, spatial_sigma)
    
    # For small images, process the whole image at once with proper padding
    if height <= 1024 and width <= 1024:
        # Add padding to avoid border issues
        padded = np.zeros((height + 2*radius, width + 2*radius, channels), dtype=np.uint8)
        
        # Copy main image content
        padded[radius:height+radius, radius:width+radius] = img
        
        # Pad borders by replicating edge pixels
        # Top and bottom
        padded[:radius, radius:-radius] = img[0:1].repeat(radius, axis=0)
        padded[-radius:, radius:-radius] = img[-1:].repeat(radius, axis=0)
        
        # Left and right
        padded[:, :radius] = padded[:, radius:radius+1].repeat(radius, axis=1)
        padded[:, -radius:] = padded[:, -radius-1:-radius].repeat(radius, axis=1)
        
        # Process the padded image
        result_padded = process_tile(padded, spatial_weights, color_sigma, radius)
        
        # Extract the result without padding
        result = result_padded[radius:-radius, radius:-radius]
        
    else:
        # For larger images, use tiling with overlap
        result = np.zeros_like(img, dtype=np.uint8)
        tile_size = 512  # Base tile size
        overlap = radius * 2
        effective_tile = tile_size - overlap  # Size of non-overlapping region
        
        # Calculate number of tiles needed
        n_tiles_y = math.ceil(height / effective_tile)
        n_tiles_x = math.ceil(width / effective_tile)
        
        # Process each tile
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Calculate tile bounds with overlap
                y_start = ty * effective_tile
                x_start = tx * effective_tile
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                
                # Create padded tile
                pad_y_before = min(radius, y_start)
                pad_y_after = min(radius, height - y_end)
                pad_x_before = min(radius, x_start)
                pad_x_after = min(radius, width - x_end)
                
                # Extract tile with padding
                tile = img[max(0, y_start - radius):y_end + pad_y_after,
                          max(0, x_start - radius):x_end + pad_x_after]
                
                # Process tile
                processed = process_tile(tile, spatial_weights, color_sigma, radius)
                
                # Calculate valid region (remove padding)
                valid_y_start = radius if y_start > 0 else 0
                valid_x_start = radius if x_start > 0 else 0
                valid_y_end = processed.shape[0] - radius if y_end < height else processed.shape[0]
                valid_x_end = processed.shape[1] - radius if x_end < width else processed.shape[1]
                
                # Copy valid region to result
                result_y_start = y_start if y_start == 0 else y_start + radius
                result_x_start = x_start if x_start == 0 else x_start + radius
                result[result_y_start:y_end, result_x_start:x_end] = processed[
                    valid_y_start:valid_y_end,
                    valid_x_start:valid_x_end
                ]
    
    return result

# ---------------------------------------------------------
# Denoising Function (Bilateral Gaussian Filter - Optimized for Speed)
# ---------------------------------------------------------

def denoise(image, edge_threshold=30, iterations=1):
    """
    Apply denoising using a bilateral Gaussian filter (edge-preserving).
    Optimized for large images like those from modern smartphones.
    
    Parameters:
    - image: Input PIL image
    - edge_threshold: Controls edge sensitivity (higher preserves more detail, default: 30)
    - iterations: Number of filter iterations (fewer preserves more detail, default: 1)
    """
    # Check image size to determine processing strategy
    width, height = image.size
    original_size = (width, height)
    img_array = np.array(image)
    
    # Set parameters based on image size and edge threshold
    color_sigma = float(edge_threshold)  # Use edge_threshold directly for more intuitive control
    spatial_sigma = 2.0
    
    # Determine if this is an extremely large image
    extremely_large_image = width * height > 12_000_000
    downsampled = False
    
    try:
        # For extremely large images, use downsampling to improve performance
        if extremely_large_image:
            print(f"Processing very large image ({width}x{height}) - using downsampling strategy")
            downsampled = True
            scale_factor = 0.5
            small_width = int(width * scale_factor)
            small_height = int(height * scale_factor)
            small_image = image.resize((small_width, small_height), Image.Resampling.LANCZOS)
            img_array = np.array(small_image)
            
        # Apply bilateral filter
        print(f"Starting denoising with edge_threshold={edge_threshold}, iterations={iterations}")
        result = img_array.copy()
        for i in range(iterations):
            print(f"Applying bilateral filter, iteration {i+1}/{iterations}")
            result = fast_bilateral_filter(
                result, 
                spatial_sigma=spatial_sigma, 
                color_sigma=color_sigma,
                use_smaller_window=(width > 1000 or height > 1000)
            )
        
        # Upscale back to original size if downsampled
        if downsampled:
            print(f"Resizing result back to original dimensions {original_size}")
            result_img = Image.fromarray(result)
            result_img = result_img.resize(original_size, Image.Resampling.LANCZOS)
            return result_img
        
        # Convert back to PIL Image
        return Image.fromarray(result)
        
    except Exception as e:
        print(f"Error in denoise function: {str(e)}")
        # Fallback to a simpler, more reliable approach if the optimized method fails
        print("Using fallback method for denoising")
        
        # Apply a gentle blur and then use PIL's built-in median filter for denoising
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        median_filtered = blurred.filter(ImageFilter.MedianFilter(size=3))
        
        # For higher edge thresholds (more detail preservation), mix with original
        if edge_threshold > 50:
            # Create a blend of original and filtered image
            blend_factor = min(0.8, edge_threshold / 100.0)
            
            # Convert to numpy arrays for blending
            original_array = np.array(image).astype(np.float32)
            filtered_array = np.array(median_filtered).astype(np.float32)
            
            # Blend the images
            result_array = (original_array * blend_factor + 
                           filtered_array * (1.0 - blend_factor)).astype(np.uint8)
            return Image.fromarray(result_array)
        
        return median_filtered

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