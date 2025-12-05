from PIL import Image



def kernel(image, kernel_size_width, kernel_size_height):
    # manual kernel implementation
    original_size_width, original_size_height = image.size

    gray_image = image.convert('L')
    pixels = gray_image.load()

    new_width = original_size_width // kernel_size_width
    new_height = original_size_height // kernel_size_height

    # Create a new image to store the result of the pooling operation
    pooled_image = Image.new('L', (new_width, new_height))
    pooled_pixels = pooled_image.load()

    for out_y in range(new_height):
        for out_x in range(new_width):
            x = out_x * kernel_size_width
            y = out_y * kernel_size_height
            # Collect all the pixel values in the 2x2 patch
            patch_values = []
            for ky in range(kernel_size_height):
                for kx in range(kernel_size_width):
                    px = x + kx
                    py = y + ky
                    if px < original_size_width and py < original_size_height:
                        patch_values.append(pixels[px, py])
            # For max-pooling, assign the maximum value in the patch to the new image
            pooled_pixels[out_x, out_y] = max(patch_values)

    pooled_image.show()

    return pooled_image

image = Image.open("images/linkedin.jpg")
kernel(image, 10, 10)

