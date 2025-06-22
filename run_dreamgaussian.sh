#!/bin/bash
# filepath: /home/khoahd/vdt_phase1/run_dreamgaussian.sh

# Directory containing PNG files
IMG_DIR="/home/khoahd/vdt_phase1/data/img_2d"
# DreamGaussian directory
DG_DIR="/home/khoahd/vdt_phase1/dreamgaussian"

# Check if image directory exists
if [ ! -d "$IMG_DIR" ]; then
    echo "Error: Image directory $IMG_DIR does not exist"
    exit 1
fi

# Check if DreamGaussian directory exists
if [ ! -d "$DG_DIR" ]; then
    echo "Error: DreamGaussian directory $DG_DIR does not exist"
    exit 1
fi

# Find all PNG files in the image directory
png_files=$(find "$IMG_DIR" -name "*.png")

# Process each PNG file
for png_file in $png_files; do
    # Extract the base filename without extension
    filename=$(basename "$png_file" .png)
    echo "Processing $filename..."
    
    # Change to DreamGaussian directory
    cd "$DG_DIR" || { echo "Failed to cd to $DG_DIR"; exit 1; }
    
    echo "Running gaussian training stage for $filename..."
    # Run the gaussian training stage
    python main.py --config configs/image_sai.yaml input=/home/khoahd/vdt_phase1/data/img_2d/${filename}.png save_path=/home/khoahd/vdt_phase1/outputs/dreamgaussian/${filename}
    
    echo "Running mesh training stage for $filename..."
    # Run the mesh training stage
    python main2.py --config configs/image_sai.yaml input=/home/khoahd/vdt_phase1/data/img_2d/${filename}.png save_path=/home/khoahd/vdt_phase1/outputs/dreamgaussian/${filename} mesh=/home/khoahd/vdt_phase1/outputs/dreamgaussian/${filename}_mesh.obj
    
    echo "Completed processing $filename"
    echo "----------------------------------------"
    
    # Go back to the original directory
    cd - > /dev/null
done

echo "All PNG files processed successfully"