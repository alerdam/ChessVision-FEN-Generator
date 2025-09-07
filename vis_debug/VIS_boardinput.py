import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# --- Main Functions ---

def start():
    """
    Clears the console and prints a stylized start message for a clean start.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n S T A R T \n")

def compute_slopes(lines_array):
    """
    Computes the slope for each line in the array to classify its orientation.
    Identifies 'vertical' and 'horizontal' lines separately.
    Slopes for other lines are rounded for easier comparison.
    
    Args:
        lines_array (np.ndarray): An array of line segments.
        
    Returns:
        list: A list of slope values or orientation strings.
    """
    slopes = []
    for line in lines_array:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            slopes.append('vertical')
        elif dy == 0:
            slopes.append('horizontal')
        else:
            slope = dy / dx
            slopes.append(round(slope, 2))
    return slopes

def detect_lines(edge_img, min_length_ratio=0.6, threshold=100, max_gap=10):
    """
    Uses the Probabilistic Hough Line Transform to detect line segments.
    
    Args:
        edge_img (np.ndarray): The Canny edge-detected image.
        min_length_ratio (float): The minimum line length as a ratio of the image width.
        threshold (int): The accumulator threshold parameter.
        max_gap (int): The maximum gap between line segments to be considered a single line.
        
    Returns:
        np.ndarray: An array of detected lines, or an empty array if none are found.
    """
    img_width = edge_img.shape[1]
    min_line_length = int(img_width * min_length_ratio)
    lines = cv2.HoughLinesP(
        edge_img,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_gap
    )
    return np.array(lines) if lines is not None else np.empty((0, 1, 4), dtype=np.int32)

def classify_lines_by_orientation(lines, slopes):
    """
    Classifies lines into vertical, horizontal, and diagonal categories based on their slopes.
    
    Args:
        lines (np.ndarray): The array of detected lines.
        slopes (list): The list of slopes for each line.
        
    Returns:
        tuple: Lists of vertical, horizontal, and diagonal lines.
    """
    vertical_lines = []
    horizontal_lines = []
    diagonal_lines = []
    for i, line in enumerate(lines):
        slope = slopes[i]
        if slope == 'vertical':
            vertical_lines.append(line)
        elif slope == 'horizontal':
            horizontal_lines.append(line)
        else:
            diagonal_lines.append((line, slope))
    print(f"\nLine Classification:")
    print(f" • Vertical lines: {len(vertical_lines)}")
    print(f" • Horizontal lines: {len(horizontal_lines)}")
    print(f" • Diagonal lines: {len(diagonal_lines)}")
    return vertical_lines, horizontal_lines, diagonal_lines

def filter_lines_by_length(lines, tolerance=10):
    """
    Filters lines based on a mode length, keeping only consistently long lines.
    
    Args:
        lines (list): The list of lines to filter.
        tolerance (int): The allowed deviation from the mode length.
        
    Returns:
        list: The list of filtered lines.
    """
    lengths = [int(np.hypot(line[0][2] - line[0][0], line[0][3] - line[0][1])) for line in lines]
    if not lengths:
        return []
    
    mode_length = Counter(lengths).most_common(1)[0][0]
    print(f"\nMode of line lengths: {mode_length}")
    
    filtered = [line for i, line in enumerate(lines) if abs(lengths[i] - mode_length) <= tolerance]
    print(f"Lines kept after length filtering: {len(filtered)}")
    return filtered

def compute_distances(lines, orientation):
    """
    Calculates the center positions of lines and the distances between them.
    
    Args:
        lines (list): The list of lines.
        orientation (str): 'horizontal' or 'vertical'.
        
    Returns:
        tuple: A tuple containing the sorted lines and the list of distances.
    """
    center_positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if orientation == 'horizontal':
            center_positions.append((y1 + y2) // 2)
        elif orientation == 'vertical':
            center_positions.append((x1 + x2) // 2)
    
    sorted_indices = np.argsort(center_positions)
    sorted_lines = [lines[i] for i in sorted_indices]
    sorted_positions = [center_positions[i] for i in sorted_indices]
    
    distances = [abs(sorted_positions[i + 1] - sorted_positions[i]) for i in range(len(sorted_positions) - 1)]
    return sorted_lines, distances

def filter_lines_to_grid(sorted_lines, distances, tolerance=2):
    """
    Filters lines to find those that form a consistent grid pattern.
    
    Args:
        sorted_lines (list): Lines sorted by their center position.
        distances (list): Distances between adjacent lines.
        tolerance (int): The allowed deviation from the mode distance.
        
    Returns:
        list: The list of lines that are part of the grid.
    """
    if not distances:
        return []
        
    mode_val = Counter(distances).most_common(1)[0][0]
    print(f"\nMode of distances: {mode_val}")
    
    filtered = [sorted_lines[0]]
    for i in range(1, len(sorted_lines)):
        if abs(distances[i - 1] - mode_val) <= tolerance:
            filtered.append(sorted_lines[i])
            
    print(f"Lines kept for grid: {len(filtered)}")
    return filtered

def detect_board_bounds(lines_list):
    """
    Calculates a bounding box (min_x, max_x, min_y, max_y) from a list of lines.
    
    Args:
        lines_list (list): A list of line segments.
        
    Returns:
        tuple: The coordinates of the bounding box.
    """
    if not lines_list:
        raise ValueError("No lines provided for bounding box detection.")
    
    coords = [(x, y) for line in lines_list for x, y in [(line[0][0], line[0][1]), (line[0][2], line[0][3])]]
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    
    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    
    print(f"\nBounding Box from filtered lines:")
    print(f" • min_x: {min_x}")
    print(f" • max_x: {max_x}")
    print(f" • min_y: {min_y}")
    print(f" • max_y: {max_y}")
    return min_x, max_x, min_y, max_y

def crop_and_save_squares(image, folder_path):
    """
    Crops the image into 8x8 squares and saves each square resized to 64x64.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    
    os.makedirs(folder_path, exist_ok=True)
    height, width = image.shape[:2]
    square_height = height // 8
    square_width = width // 8

    if square_height == 0 or square_width == 0:
        raise ValueError("Cropped image is too small to divide into 8x8 squares.")
    
    for row in range(8):
        for col in range(8):
            y_start, y_end = row * square_height, (row + 1) * square_height
            x_start, x_end = col * square_width, (col + 1) * square_width
            square_roi = image[y_start:y_end, x_start:x_end]

            # Resize to 64x64
            resized_square = cv2.resize(square_roi, (64, 64), interpolation=cv2.INTER_AREA)

            filename = os.path.join(folder_path, f"square_{row}_{col}.jpg")
            cv2.imwrite(filename, resized_square)
            
    print(f"All squares saved in '{folder_path}' folder.")

def visualize_squares(image):
    """
    Displays all 64 individual chessboard squares in a single figure.
    The layout is adjusted to have a little more space between squares.
    
    Args:
        image (np.ndarray): The cropped board image.
    """
    if image is None or image.size == 0:
        print("Cropped image is empty, cannot visualize squares.")
        return
        
    square_size = int(image.shape[0] / 8)
    
    plt.figure(figsize=(10, 10))
    plt.suptitle("Squares", fontsize=20, y=0.98)
    
    for i in range(8):
        for j in range(8):
            y_start, x_start = i * square_size, j * square_size
            square_roi = image[y_start:y_start + square_size, x_start:x_start + square_size]
            
            plt.subplot(8, 8, i * 8 + j + 1)
            plt.imshow(square_roi, cmap='gray', vmin=0, vmax=255)
            plt.title(f"({i}, {j})", fontsize=10)
            plt.axis('off')
            
    # Add spacing
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

# --- Main Execution Flow ---
def process_image(img_path):
    """
    Main function to process the image and extract chessboard squares with
    visual debugging.
    
    Args:
        img_path (str): The path to the input image.
    """
    start()
    
    # Step 1: Load and process the image
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img_height, img_width = gray_img.shape
    print(f"Image Height: {img_height} | Image Width: {img_width}")

    # Step 2: Edge Detection & Hough Line Transform
    edges_img = cv2.Canny(gray_img, 100, 200)
    lines_array = detect_lines(edges_img, min_length_ratio=0.5)
    if lines_array.size == 0:
        print("No lines were detected. Exiting.")
        return

    # Step 3: Classify and Filter Lines
    slopes = compute_slopes(lines_array)
    print(f"Number of total lines detected: {len(lines_array)}\n")
    vertical_lines, horizontal_lines, _ = classify_lines_by_orientation(lines_array, slopes)
    
    filtered_horiz = filter_lines_by_length(horizontal_lines)
    filtered_vert = filter_lines_by_length(vertical_lines)
    
    if len(filtered_horiz) < 2 or len(filtered_vert) < 2:
        print("Not enough lines detected for filtering. Skipping cropping.")
        return
        
    sorted_horiz, horiz_dists = compute_distances(filtered_horiz, 'horizontal')
    sorted_vert, vert_dists = compute_distances(filtered_vert, 'vertical')
    
    filtered_horiz_grid = filter_lines_to_grid(sorted_horiz, horiz_dists)
    filtered_vert_grid = filter_lines_to_grid(sorted_vert, vert_dists)
    
    if len(filtered_horiz_grid) < 2 or len(filtered_vert_grid) < 2:
        print("Not enough grid lines found to define a bounding box. Skipping cropping.")
        return

    # Step 4: Find Bounding Box and Crop
    combined_filtered_lines = filtered_horiz_grid + filtered_vert_grid
    Board_min_x, Board_max_x, Board_min_y, Board_max_y = detect_board_bounds(combined_filtered_lines)
    
    cropped_board = gray_img[Board_min_y:Board_max_y, Board_min_x:Board_max_x]

    # Step 5: Save cropped board and squares
    cropped_board_path = 'workfiles/CroppedBoard.jpg'
    os.makedirs(os.path.dirname(cropped_board_path), exist_ok=True)
    cv2.imwrite(cropped_board_path, cropped_board)
    print(f"\nCropped board saved as '{cropped_board_path}'")
    
    squares_folder_path = 'workfiles/boardSquares'
    try:
        crop_and_save_squares(cropped_board, squares_folder_path)
    except ValueError as e:
        print(f"Error during square cropping: {e}")
        return

    # Step 6: Visual Debug 
    print("\n\nCreating Visual Debug Study...")
    
    # Visualization 1: Original Image with Annotations
    annotated_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
    for line in lines_array:
        x1, y1, x2, y2 = line[0]
        cv2.line(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    for line in combined_filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 8)
    
    cv2.rectangle(annotated_img, (Board_min_x, Board_min_y), (Board_max_x, Board_max_y), (0, 0, 255), 5)
    
    plt.figure(figsize=(15, 7))
    plt.suptitle("Board Detection", fontsize=18)
    
    plt.subplot(1, 2, 1)
    plt.imshow(edges_img, cmap='gray')
    plt.title('Canny Edge Detection', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Grid Lines & Bounding Box', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Visualization 2: Final Cropped Board
    plt.figure(figsize=(7, 7))
    plt.suptitle("Board", fontsize=18)
    plt.imshow(cropped_board, cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Visualization 3: All 64 Squares
    visualize_squares(cropped_board)

# --- Main Execution ---
if __name__ == "__main__":
    imgPath = "boards/test_2.jpg"
    try:
        process_image(imgPath)
    except Exception as e:
        print(f"An error occurred during image processing: {e}")