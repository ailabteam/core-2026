"""
Configuration constants for Document Image Enhancement Pipeline
"""

# Target document dimensions (A4 at 300 DPI)
TARGET_DPI = 300
TARGET_HEIGHT = 2480  # A4 height at 300 DPI (210mm)
TARGET_WIDTH = 1754   # A4 width at 300 DPI (297mm)

# Processing modes
MODE_AI = 'ai'          # For modern AI models (DeepSeek, Gemini, etc.) - natural, human-readable
MODE_OCR = 'ocr'        # For classical OCR - binary, high contrast

# Image processing parameters (AI-friendly - moderate)
MEDIAN_KERNEL_SIZE = 3      # For noise reduction (light)
BILATERAL_D = 5             # Bilateral filter diameter (reduced for natural look)
BILATERAL_SIGMA_COLOR = 50  # Bilateral filter sigma color (reduced)
BILATERAL_SIGMA_SPACE = 50  # Bilateral filter sigma space (reduced)

# Contrast enhancement (CLAHE) - moderate for AI
CLAHE_CLIP_LIMIT = 1.5      # Contrast limiting (reduced for natural look)
CLAHE_TILE_GRID_SIZE = (8, 8)  # Grid size for CLAHE

# Edge detection (Canny)
CANNY_THRESHOLD1 = 50       # Lower threshold
CANNY_THRESHOLD2 = 150      # Upper threshold
CANNY_APERTURE = 3          # Aperture size for Sobel operator

# Adaptive thresholding (OCR mode only)
ADAPTIVE_BLOCK_SIZE = 11    # Block size for adaptive threshold
ADAPTIVE_C = 2              # Constant subtracted from mean

# Hough Line Transform (for deskew) - gentler
HOUGH_RHO = 1               # Distance resolution in pixels
HOUGH_THETA = 1             # Angle resolution in degrees (converted to radians)
HOUGH_THRESHOLD = 80        # Accumulator threshold (reduced for gentler detection)
HOUGH_MIN_LINE_LENGTH = 80  # Minimum line length (reduced)
HOUGH_MAX_LINE_GAP = 15     # Maximum gap between line segments (increased for tolerance)

# Contour detection
CONTOUR_EPSILON_FACTOR = 0.02  # Approximation accuracy factor
MIN_CONTOUR_AREA = 10000       # Minimum contour area to consider

# Morphological operations
MORPH_KERNEL_SIZE = (3, 3)  # Kernel size for morphological ops
MORPH_ITERATIONS = 1        # Number of iterations

# Quality thresholds
MIN_QUALITY_SCORE = 0.3     # Minimum acceptable quality score
MAX_SKEW_ANGLE = 10         # Maximum correction angle (reduced for gentler rotation)

# Processing options
DEFAULT_GRAYSCALE = False   # Keep color by default for AI models
PRESERVE_ASPECT_RATIO = True  # Maintain aspect ratio when resizing
DEFAULT_MODE = MODE_AI      # Default to AI-friendly processing

# Sharpening parameters (light for AI)
SHARPEN_AMOUNT = 0.3        # Very light sharpening
SHARPEN_SIGMA = 1.0         # Gaussian sigma for unsharp mask
SHARPEN_KERNEL_SIZE = (5, 5)  # Kernel size
