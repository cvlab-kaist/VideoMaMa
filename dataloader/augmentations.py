import random
import cv2
import numpy as np
from PIL import Image, ImageDraw


def augment_to_bounding_box(mask_image):
    """Converts a mask to its bounding box representation."""
    mask_np = np.array(mask_image)
    points = cv2.findNonZero(mask_np)
    if points is None:
        return Image.new('L', mask_image.size, 0)
    x, y, w, h = cv2.boundingRect(points)
    new_mask = Image.new('L', mask_image.size, 0)
    draw = ImageDraw.Draw(new_mask)
    draw.rectangle([(x, y), (x + w, y + h)], fill=255)
    return new_mask


def augment_to_polygon(mask_image, simplification_tolerance):
    """
    Converts a mask to a simplified polygon and back to a mask.
    The level of simplification is controlled by `simplification_tolerance`.
    """
    mask_np = np.array(mask_image)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.new('L', mask_image.size, 0)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = simplification_tolerance * cv2.arcLength(largest_contour, True)
    approximated_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    new_mask = Image.new('L', mask_image.size, 0)
    draw = ImageDraw.Draw(new_mask)
    squeezed_points = approximated_polygon.squeeze()
    if squeezed_points.ndim == 1:
        polygon_points = [tuple(squeezed_points)]
    else:
        polygon_points = [tuple(point) for point in squeezed_points]
    if len(polygon_points) > 1:
        draw.polygon(polygon_points, fill=255)
    return new_mask


def augment_with_instability(mask_image, noise_level=0.05, salt_vs_pepper_ratio=0.5):
    """
    Introduces instability to a mask by adding salt-and-pepper noise to imitate
    segmentation artifacts like holes and stray pixels.

    Args:
        mask_image (PIL.Image.Image): The input binary mask image.
        noise_level (float): The proportion of pixels in the bounding box to be affected.
        salt_vs_pepper_ratio (float): The ratio of salt (white pixels) to pepper (black pixels).
                                     0.5 means an equal amount of both.

    Returns:
        PIL.Image.Image: The augmented mask with noise.
    """
    if noise_level == 0:
        return mask_image

    # Convert PIL image to numpy array for manipulation
    mask_np = np.array(mask_image)

    # Find the bounding box of the mask to apply noise only in the relevant area
    points = cv2.findNonZero(mask_np)
    if points is None:
        return mask_image  # Return original image if mask is empty

    x, y, w, h = cv2.boundingRect(points)

    # Calculate the number of noisy pixels to add
    bbox_area = w * h
    num_noise_pixels = int(bbox_area * noise_level)

    # Calculate the number of salt and pepper pixels
    num_salt = int(num_noise_pixels * salt_vs_pepper_ratio)
    num_pepper = num_noise_pixels - num_salt

    # Add salt (white pixels)
    # These can appear anywhere in the bounding box, creating stray white dots
    salt_coords = [
        np.random.randint(y, y + h, num_salt),
        np.random.randint(x, x + w, num_salt)
    ]
    mask_np[salt_coords[0], salt_coords[1]] = 255

    # Add pepper (black pixels)
    # These are only effective where the mask is already white, creating holes
    pepper_coords = [
        np.random.randint(y, y + h, num_pepper),
        np.random.randint(x, x + w, num_pepper)
    ]
    mask_np[pepper_coords[0], pepper_coords[1]] = 0

    # Convert back to PIL image and return
    return Image.fromarray(mask_np, mode='L')


def augment_to_polygon_with_nested_holes(mask_image, simplification_tolerance):
    """
    내부에 중첩된 구멍과 도형까지 모두 완벽하게 보존하는 다각형 마스크를 생성합니다.
    """
    mask_np = np.array(mask_image)

    # RETR_TREE 모드를 사용하여 모든 윤곽선과 계층 정보를 얻습니다.
    contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return Image.new('L', mask_image.size, 0)

    # 각 윤곽선의 깊이를 저장할 리스트를 생성합니다.
    depths = [-1] * len(contours)

    # ⭐️ 수정된 함수: 재귀 대신 반복을 사용해 깊이를 계산합니다.
    def calculate_depth_iterative(root_index, current_depth):
        current_idx = root_index
        while current_idx != -1:
            if depths[current_idx] == -1:
                depths[current_idx] = current_depth

                child_idx = hierarchy[0][current_idx][2]
                if child_idx != -1:
                    # 자식은 재귀적으로 호출하여 깊이를 늘립니다.
                    calculate_depth_iterative(child_idx, current_depth + 1)

            # 다음 형제 윤곽선으로 이동합니다.
            current_idx = hierarchy[0][current_idx][0]

    # 최상위 윤곽선(parent = -1)을 찾아 깊이 계산을 시작합니다.
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            calculate_depth_iterative(i, 0)

    new_mask = Image.new('L', mask_image.size, 0)
    draw = ImageDraw.Draw(new_mask)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 4:
            continue

        epsilon = simplification_tolerance * cv2.arcLength(contour, True)
        approximated_polygon = cv2.approxPolyDP(contour, epsilon, True)

        if approximated_polygon.shape[0] >= 3:
            squeezed_points = approximated_polygon.squeeze(axis=1)
            polygon_points = [tuple(point) for point in squeezed_points]

            # 계산된 깊이를 기준으로 색상을 결정합니다.
            fill_color = 255 if depths[i] % 2 == 0 else 0
            draw.polygon(polygon_points, fill=fill_color)

    return new_mask

def augment_to_polygon_preserve_all_parts(mask_image, simplification_tolerance):
    """
    Converts all parts of a mask to simplified polygons, preserving all
    disconnected components. The level of simplification is controlled by
    `simplification_tolerance`.
    """
    # Convert the PIL image to a NumPy array for OpenCV processing
    mask_np = np.array(mask_image)

    # Find all external contours in the mask
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return an empty image
    if not contours:
        return Image.new('L', mask_image.size, 0)

    # Create a new blank mask to draw our polygons on
    new_mask = Image.new('L', mask_image.size, 0)
    draw = ImageDraw.Draw(new_mask)

    # ⭐️ KEY CHANGE: Loop through every contour found, not just the largest one
    for contour in contours:
        # Optional: you can ignore very small contours that might be noise
        if cv2.contourArea(contour) < 4:
            continue

        # Calculate the simplification tolerance for each contour individually
        epsilon = simplification_tolerance * cv2.arcLength(contour, True)
        approximated_polygon = cv2.approxPolyDP(contour, epsilon, True)

        # The polygon points need to be in a list of tuples format for drawing
        # We also ensure the polygon has at least 3 vertices to be a valid shape
        if approximated_polygon.shape[0] >= 3:
            # Squeeze the array from (num_points, 1, 2) to (num_points, 2)
            squeezed_points = approximated_polygon.squeeze(axis=1)
            # Convert the NumPy points to a list of tuples
            polygon_points = [tuple(point) for point in squeezed_points]
            # Draw the resulting polygon on the new mask
            draw.polygon(polygon_points, fill=255)

    return new_mask


def augment_by_resizing(mask_image, downsample_factor, binary_mask_threshold=127):
    """Downsamples and upsamples the mask to remove fine details."""
    original_size = mask_image.size
    small_size = (original_size[0] // downsample_factor, original_size[1] // downsample_factor)
    downsampled = mask_image.resize(small_size, Image.Resampling.BILINEAR)
    upsampled = downsampled.resize(original_size, Image.Resampling.BILINEAR)
    return upsampled.point(lambda p: 255 if p > binary_mask_threshold else 0, mode='L')


def apply_all_augmentations(mask_image, downsample_factor, simplification_tolerance, instability_noise_level=0.05):
    """Applies a random sequence of available augmentations."""
    augmentations = [
        lambda m: augment_to_polygon_with_nested_holes(m, simplification_tolerance),
        lambda m: augment_by_resizing(m, downsample_factor),
        # lambda m: augment_with_instability(m, noise_level=instability_noise_level),
    ]
    random.shuffle(augmentations)
    num_to_apply = random.randint(1, len(augmentations))
    augmented_mask = mask_image
    for i in range(num_to_apply):
        augmented_mask = augmentations[i](augmented_mask)
    return augmented_mask


def augment_with_temporal_occlusion(mask_frames, num_occlusions, occlusion_shape, occlusion_scale_range, kernel_size=5):
    """
    Applies a diverse set of temporal augmentations to randomly selected mask frames.
    For each frame selected for augmentation, one of the following operations is chosen randomly:
    1. Occlusion: The original method of adding a shape to hide part of the mask.
    2. None Mask: Replaces the mask with a completely empty (black) frame.
    3. All Mask: Replaces the mask with a completely full (white) frame.
    4. Erosion: Erodes the mask boundaries.
    5. Dilation: Dilates the mask boundaries.
    """
    if not mask_frames:
        return mask_frames

    new_mask_frames = list(mask_frames)
    indices_to_augment = random.choices(range(len(mask_frames)), k=num_occlusions)

    for idx in indices_to_augment:
        original_mask = new_mask_frames[idx]

        # Define the pool of possible augmentation operations for each frame
        def occlude_mask(mask):
            occluded_mask = mask.copy()
            draw = ImageDraw.Draw(occluded_mask)
            mask_np = np.array(occluded_mask)
            points = cv2.findNonZero(mask_np)
            if points is None: return mask

            x, y, w, h = cv2.boundingRect(points)
            current_scale = random.uniform(occlusion_scale_range[0], occlusion_scale_range[1])
            occlusion_w, occlusion_h = int(w * current_scale), int(h * current_scale)
            max_offset_x, max_offset_y = w - occlusion_w, h - occlusion_h
            offset_x = random.randint(0, max_offset_x) if max_offset_x > 0 else 0
            offset_y = random.randint(0, max_offset_y) if max_offset_y > 0 else 0
            occlusion_x, occlusion_y = x + offset_x, y + offset_y

            if occlusion_shape == 'rectangle':
                draw.rectangle([(occlusion_x, occlusion_y), (occlusion_x + occlusion_w, occlusion_y + occlusion_h)],
                               fill=0)
            elif occlusion_shape == 'circle':
                draw.ellipse([(occlusion_x, occlusion_y), (occlusion_x + occlusion_w, occlusion_y + occlusion_h)],
                             fill=0)
            return occluded_mask

        def none_mask(mask):
            return Image.new('L', mask.size, 0)

        def all_mask(mask):
            return Image.new('L', mask.size, 255)

        def erode_mask(mask):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = np.array(mask)
            eroded_np = cv2.erode(mask_np, kernel, iterations=1)
            return Image.fromarray(eroded_np, mode='L')

        def dilate_mask(mask):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = np.array(mask)
            dilated_np = cv2.dilate(mask_np, kernel, iterations=1)
            return Image.fromarray(dilated_np, mode='L')

        augmentation_choices = [occlude_mask, none_mask, all_mask, erode_mask, dilate_mask]

        # Randomly select and apply one augmentation to the current frame
        chosen_augmentation = random.choice(augmentation_choices)
        new_mask_frames[idx] = chosen_augmentation(original_mask)

    return new_mask_frames