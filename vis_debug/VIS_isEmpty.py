import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def is_square_empty(
    image_path,
    inner_ratio=0.3,
    outer_ratio=0.52,
    brightness_threshold=5,
):
    """
    Determines if a chessboard square is empty based on simple brightness contrast.
    
    Args:
        image_path (str): Path to the square image.
        inner_ratio (float): Radius of the inner circle as a ratio of image size.
        outer_ratio (float): Radius of the outer circle as a ratio of image size.
        brightness_threshold (int): The threshold for mean brightness difference.
        
    Returns:
        bool: True if the square is empty, False otherwise.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    r_center = max(2, int(min(h, w) * inner_ratio))
    r_outer = max(r_center + 2, int(min(h, w) * outer_ratio))

    center_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(center_mask, (cx, cy), r_center, 255, -1)
    outer_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(outer_mask, (cx, cy), r_outer, 255, -1)
    background_mask = cv2.bitwise_xor(outer_mask, center_mask)

    mean_center = cv2.mean(gray, mask=center_mask)[0]
    mean_bg = cv2.mean(gray, mask=background_mask)[0]
    brightness_diff = abs(mean_center - mean_bg)

    # If the brightness difference is below the threshold, the square is likely empty.
    is_empty = brightness_diff < brightness_threshold
    
    return is_empty

def visualize_square(image_path, is_empty):
    """
    Shows a visualization plot for a single square.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    r_center = max(2, int(min(h, w) * 0.1))
    r_outer = max(r_center + 2, int(min(h, w) * 0.52))
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    cv2.circle(vis, (cx, cy), r_center, (0, 255, 0), 2)
    cv2.circle(vis, (cx, cy), r_outer, (0, 0, 255), 2)
    
    return vis

def run_is_empty_debug(directory):
    """
    Loads all square images from a directory, classifies them as empty or not,
    and displays the results in a compact 8x8 grid.
    """
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.suptitle("is_square_empty", fontsize=16, y=1.0)
    
    for r in range(8):
        for c in range(8):
            file_path = os.path.join(directory, f"square_{r}_{c}.jpg")
            ax = axes[r, c]

            if os.path.exists(file_path):
                try:
                    is_empty = is_square_empty(file_path)
                    vis = visualize_square(file_path, is_empty)
                    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                    title = "Empty" if is_empty else "Not Empty"
                    ax.set_title(title, fontsize=10, color='green' if is_empty else 'red')
                except Exception as e:
                    ax.set_title(f"Error\n{e}", color='red', fontsize=8)
            else:
                ax.set_title("File not found", color='red', fontsize=8)
            
            ax.axis('off')
            
    # Adjust subplot parameters to make the grid tighter and prevent labels from being cut off
    plt.subplots_adjust(wspace=0.05, hspace=0.1) # Reduced space between squares
    plt.tight_layout() # Adjusts subplot params for tight layout
    plt.show()

# Main Execution Block
if __name__ == "__main__":
    squares_folder = "workfiles/boardSquares"
    run_is_empty_debug(squares_folder)