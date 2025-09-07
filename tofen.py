import cv2
import numpy as np
import os
import tensorflow as tf
import BoardInput

# Config
COLOR_MODEL_PATH = "siyahbeyaz.keras"
PIECE_MODEL_PATH = "pieces.keras"
IMG_SIZE = 64
PIECE_CLASS_NAMES = ['b', 'k', 'n', 'p', 'q', 'r']
COLOR_CLASS_NAMES = ['black', 'white']

# Load Models
if not os.path.exists(COLOR_MODEL_PATH):
    raise FileNotFoundError(f"Color model not found: {COLOR_MODEL_PATH}")
color_model = tf.keras.models.load_model(COLOR_MODEL_PATH)

if not os.path.exists(PIECE_MODEL_PATH):
    raise FileNotFoundError(f"Piece model not found: {PIECE_MODEL_PATH}")
piece_model = tf.keras.models.load_model(PIECE_MODEL_PATH)

def start():
    """Clears the console and prints a start message."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n S T A R T \n")

def is_square_empty(
    image_path,
    inner_ratio=0.3,
    outer_ratio=0.52,
    brightness_threshold=5,
):
    """
    Determines if a chess square is empty based on a simple brightness difference.
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
    Classifies the color and type of a piece and returns the FEN notation.
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

def FEN_Generator(
    directory,
    is_From_White_Side=True,
    active_color='w',
    castling_rights='-',
    en_passant='-',
    halfmove_clock=0,
    fullmove_number=1
):
    """
    Classifies all squares and generates the FEN notation based on board orientation.
    
    Args:
        directory (str): The path to the folder containing the square images.
        is_white_side_up (bool): Whether the board is oriented with the white side up.
        
    Returns:
        str: The FEN notation for the board.
    """
    # Load board data into an 8x8 matrix from the images
    board_status = [['' for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            file_path = os.path.join(directory, f"square_{r}_{c}.jpg")
            
            if not os.path.exists(file_path):
                board_status[r][c] = "FILE_NOT_FOUND"
                continue
            
            is_empty = is_square_empty(file_path)
            
            if is_empty:
                board_status[r][c] = '0'
            else:
                piece_info = classify_piece_and_color(file_path)
                board_status[r][c] = piece_info
    
    # Generate the FEN string in the correct reading direction
    fen_ranks = []
    
    # Read normally if the white side is up, otherwise read in reverse
    row_range = range(8) if is_From_White_Side else range(7, -1, -1)
    col_range = range(8) if is_From_White_Side else range(7, -1, -1)
    
    for r in row_range:
        empty_count = 0
        fen_rank = ""
        for c in col_range:
            piece = board_status[r][c]

            if piece == '0':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece
        
        if empty_count > 0:
            fen_rank += str(empty_count)
            
        fen_ranks.append(fen_rank)
    
    piece_placement = "/".join(fen_ranks)
    
    full_fen = f"{piece_placement} {active_color} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"
    return full_fen

# Main Execution Block
if __name__ == "__main__":
    BOARDPATH = "boards/test_1.jpg"
    squares_folder = "workfiles/boardSquares"
    start()

    BoardInput.process_image(BOARDPATH) 
    
    try:
        fen_string_white = FEN_Generator(squares_folder,
                                         is_From_White_Side=True,
                                         active_color="w",
                                         castling_rights="KQkq")
        print("\n" + "-"*30 + "\n")
        print(fen_string_white)
        print("\n" + "-"*30 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")