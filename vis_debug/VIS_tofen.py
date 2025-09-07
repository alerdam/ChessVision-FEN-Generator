import cv2
import numpy as np
import os
import csv
import re
import tensorflow as tf
import matplotlib.pyplot as plt

# Config
COLOR_MODEL_PATH = "siyahbeyaz.keras"
PIECE_MODEL_PATH = "pieces.keras"
IMG_SIZE = 64
PIECE_CLASS_NAMES = ['b', 'k', 'n', 'p', 'q', 'r']
COLOR_CLASS_NAMES = ['black', 'white']
ORIGINAL_BOARD_PATH = "workfiles/CroppedBoard.jpg" # Path to the cropped board image

# Load Models
if not os.path.exists(COLOR_MODEL_PATH):
    raise FileNotFoundError(f"Color model not found: {COLOR_MODEL_PATH}")
color_model = tf.keras.models.load_model(COLOR_MODEL_PATH)

if not os.path.exists(PIECE_MODEL_PATH):
    raise FileNotFoundError(f"Piece model not found: {PIECE_MODEL_PATH}")
piece_model = tf.keras.models.load_model(PIECE_MODEL_PATH)

def is_square_empty(
    image_path,
    inner_ratio=0.3,
    outer_ratio=0.52,
    brightness_threshold=5,
):
    """
    Determines if a chessboard square is empty based on simple brightness contrast.
    """
    if not os.path.exists(image_path):
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False
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

    is_empty = brightness_diff < brightness_threshold
    
    return is_empty

def classify_piece_and_color(image_path):
    """
    Classifies a piece's color and type and returns the FEN notation.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "ERROR_IMG_READ"

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    try:
        color_pred = color_model.predict(img_input, verbose=0)
        color_class = COLOR_CLASS_NAMES[np.argmax(color_pred)]

        piece_pred = piece_model.predict(img_input, verbose=0)
        piece_class = PIECE_CLASS_NAMES[np.argmax(piece_pred)]

        # Return FEN notation
        if color_class == 'white':
            return piece_class.upper()
        else:
            return piece_class.lower()

    except Exception as e:
        return "ERROR_PRED"

def classify_board_and_create_csv(directory, output_filename="chessboard_status.csv"):
    """
    Classifies all squares and saves the status to an 8x8 CSV file
    using FEN notation and '0' for empty squares.
    """
    chessboard_status = [['' for _ in range(8)] for _ in range(8)]
    
    for r in range(8):
        for c in range(8):
            file_path = os.path.join(directory, f"square_{r}_{c}.jpg")
            
            if not os.path.exists(file_path):
                chessboard_status[r][c] = "FILE_NOT_FOUND"
                continue
                
            is_empty = is_square_empty(file_path)
            
            if is_empty:
                chessboard_status[r][c] = '0' # CSV format for empty squares
            else:
                piece_info = classify_piece_and_color(file_path)
                chessboard_status[r][c] = piece_info

    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(chessboard_status)
    
    print(f"Chessboard status saved to {output_filename}")
    return chessboard_status

def draw_label(image, label, color):
    """
    Helper function to draw bold, centered text on an image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3 # Increased thickness for bold text
    
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Calculate the coordinates for the text to be centered
    text_x = (image.shape[1] - text_width) // 2
    text_y = (image.shape[0] + text_height) // 2
    
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)
    return image

def visualize_final_board(csv_path, original_image_path):
    """
    Visualizes the final chessboard status with FEN labels in red,
    and 'Null' for empty squares.
    """
    board_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if board_img is None:
        print(f"Error: Original board image not found at {original_image_path}")
        return

    chessboard_status = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            chessboard_status.append(row)
    
    square_size = board_img.shape[0] // 8
    
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    fig.suptitle("Final Chessboard Status", fontsize=20, y=1.0)

    for r in range(8):
        for c in range(8):
            label = chessboard_status[r][c]
            ax = axes[r, c] 
            
            y_start = r * square_size
            x_start = c * square_size
            
            square_roi = board_img[y_start : y_start + square_size, x_start : x_start + square_size]
            vis_img = cv2.cvtColor(square_roi, cv2.COLOR_GRAY2BGR)
            
            vis_label = 'Null' if label == '0' else label
            
            text_color = (0, 0, 255) # Red in BGR
            
            labeled_img = draw_label(vis_img, vis_label, text_color)
            
            ax.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


# Main Execution Block
if __name__ == "__main__":
    squares_folder = "workfiles/boardSquares"
    output_csv = "chessboard_status.csv"
    
    board_status_data = classify_board_and_create_csv(squares_folder, output_csv)

    visualize_final_board(output_csv, ORIGINAL_BOARD_PATH)