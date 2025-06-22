#!/bin/bash
# filepath: /home/khoahd/vdt_phase1/run_dreamgaussian.sh

IMG_DIR="/home/khoahd/vdt_phase1/data/img_2d"
DG_DIR="/home/khoahd/vdt_phase1/threestudio"
FINAL_OUTPUT_DIR="/home/khoahd/vdt_phase1/outputs/stable-zero123"

# Check if directories exist
if [ ! -d "$IMG_DIR" ]; then
    echo "Error: Image directory $IMG_DIR does not exist"
    exit 1
fi

if [ ! -d "$DG_DIR" ]; then
    echo "Error: DreamGaussian directory $DG_DIR does not exist"
    exit 1
fi

# Find all PNG files
png_files=$(find "$IMG_DIR" -name "*.png")

# Process each file
for png_file in $png_files; do
    filename=$(basename "$png_file" .png)
    filename_rgba="${filename}_rgba"
    echo "Processing $filename..."

    # Go to DreamGaussian dir
    cd "$DG_DIR" || { echo "Failed to cd to $DG_DIR"; exit 1; }

    echo "Running training for $filename..."
    CUDA_VISIBLE_DEVICES=5 python launch.py --config configs/stable-zero123.yaml --train --gpu 0 data.image_path="$IMG_DIR/${filename}.png"

    echo "Finding output folder..."
    echo "DEBUG: Listing all output directories:"
    ls -l threestudio/outputs/zero123-sai

    echo "DEBUG: Searching for pattern: *${filename}.png@*"
    folder=$(find /home/khoahd/vdt_phase1/threestudio/outputs/zero123-sai -maxdepth 1 -type d -name "*${filename}.png@*" | sort | tail -n 1)
    echo "DEBUG: Output folder found: $folder"

    if [ -z "$folder" ]; then
        echo "Error: No output folder found for $filename"
        exit 1
    fi

    echo "Exporting mesh from $folder..."
    CUDA_VISIBLE_DEVICES=5 python launch.py --config configs/stable-zero123.yaml --export --gpu 0 resume="${folder}/ckpts/last.ckpt" system.exporter_type=mesh-exporter

    echo "Copying mesh model..."
    EXPORT_ROOT="/home/khoahd/vdt_phase1/threestudio/outputs/zero123-sai"
    export_folder=$(find "$EXPORT_ROOT" -type d -name "*hamburger_rgba.png@*" | sort | tail -n 1)

    echo "DEBUG: Export folder FOUNDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD: $export_folder"

    if [ -z "$export_folder" ]; then
        echo "Error: No export folder found for hamburger mesh"
        exit 1
    fi

    obj_path="${export_folder}/save/it800-export/model.obj"

    if [ ! -f "$obj_path" ]; then
        echo "Error: model.obj not found in $obj_path"
        exit 1
    fi

    cp "$obj_path" "$FINAL_OUTPUT_DIR/${filename}.obj"


    echo "Done with $filename"
    echo "----------------------------------------"

    cd - > /dev/null
done

echo "All PNG files processed successfully."
