# ChessVision - FEN Generator

This project is a computer vision application that automatically recognizes the state of a chess board from an image and generates its corresponding FEN (Forsyth–Edwards Notation) string. The code is designed to identify squares, pieces, and their colors correctly using image processing and deep learning models.

## Key Features

* **FEN Generation**: Converts a chess board image into a standard FEN string.
* **Object Detection**: Distinguishes between empty and occupied squares based on a simple brightness difference.
* **Piece and Color Classification**: Employs two custom-trained TensorFlow models to accurately identify the piece type (e.g., pawn, queen) and color (black or white) on each square.


## Architecture

The project's architecture is a sequential pipeline, designed to process an image of a chessboard and convert it into a digital FEN representation.

1.  #### Board Detection and Segmentation
    This module detects the chessboard within the image and segments it into 64 individual squares for subsequent analysis.

    <img src="doc_images\BoardDetection.png" width="400" />

2.  #### Is Square Empty?
    This module determines if a chess square is occupied or empty by comparing the brightness of its center to its background. It uses a simple but effective technique to differentiate between a piece and the underlying board color.

    <img src="doc_images\isEmpty.jpg" width="400" />

3.  #### Piece and Color Classification
    This module uses two separate TensorFlow deep learning models, pieces.keras and siyahbeyaz.keras, to classify the piece and its color. It takes an individual square's image as input, predicts the piece type and color, and returns the corresponding FEN notation.

    <img src="doc_images\pieces.jpg" width="400" />

4.  #### FEN Generator
    This module is the final stage of the pipeline. It orchestrates the process of analyzing all 64 squares and assembles the results into a complete FEN string. The module correctly formats the output, handling empty squares and ensuring the board orientation (white or black side up) is properly accounted for in the final notation.

## Future Work

* **Mobile Integration**: Develop a mobile application to allow users to capture photos or upload screenshots for instant FEN generation on the go.
* **Engine Integration & Move Suggestion**: Integrate a chess engine (e.g., Stockfish) to provide "Best Move" recommendations and position analysis.
* **Unified Model Architecture**: Consolidate the piece recognition and color classification tasks into a single, multi-task deep learning model for better performance.
* **Advanced Computer Vision Methods**: Implement advanced CV techniques, including Perspective Transform for skewed images and enhanced grid detection.


## Prerequisites & Installation

### Prerequisites

To run this project, you'll need the following software and libraries installed on your system:

* **Python 3.x**
* **TensorFlow**
* **OpenCV (`cv2`)**
* **NumPy**:

### Installation & Usage

1.  **Clone the Repository**: First, get the project files by cloning the repository from GitHub:
    ```bash
    git clone https://github.com/alerdam/ChessVision-FEN-Generator.git
    cd ChessVision-FEN-Generator
    ```

2.  **Install Dependencies**: Navigate to the project directory and install all the required libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place Model Files**: Ensure the trained model files (`siyahbeyaz.keras` and `pieces.keras`) are located in the project's root directory.

