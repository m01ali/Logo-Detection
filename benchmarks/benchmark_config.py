"""Central configuration for all benchmark runs."""

# HuggingFace model IDs
MODELS = {
    "clip":    "openai/clip-vit-large-patch14",
    "siglip2": "google/siglip2-so400m-patch14-384",
    "dinov2":  "facebook/dinov2-large",
    "hybrid":  None,  # built from clip + dinov2
}

# Post-normalisation embedding dimensions
EMBEDDING_DIMS = {
    "clip":    768,
    "siglip2": 1152,
    "dinov2":  1024,
    "hybrid":  1792,  # 768 + 1024, re-L2-normalised
}

# Paths
DATABASE_PATH = "/Users/alii/Downloads/logo_for_database"
RESULTS_DIR   = "/Users/alii/Documents/Logo_Detection_App_New_20_10_25/benchmarks/results"
LABELLED_CROPS_DIR = "/Users/alii/Documents/Logo_Detection_App_New_20_10_25/benchmarks/labelled_crops"

# Retrieval settings
DEFAULT_K_VALUES    = [1, 3, 5]
THRESHOLD_SWEEP     = [round(t, 2) for t in [x / 20 for x in range(10, 20)]]  # 0.50 to 0.95 step 0.05
PRODUCTION_THRESHOLD = 0.85  # used as a reference dashed line in plots

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
