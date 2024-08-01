import os
from PIL import Image
import argparse

def convert_tiff_to_png(input_dir, output_dir=None):
    """Converts TIFF images in a directory to PNG format and saves them in another directory.
       If no output directory is provided, it defaults to a 'pngs' folder in the input directory."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory {input_dir} does not exist.")
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'pngs')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(input_dir, filename)
            with Image.open(file_path) as img:
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_dir, output_filename)
                img.save(output_path, 'PNG')
            print(f"Converted {filename} to PNG")

def main():
    parser = argparse.ArgumentParser(description="Convert TIFF images to PNG format.")
    parser.add_argument('-i', type=str, help='Directory containing TIFF images')
    parser.add_argument('-o', type=str, help='Directory to save converted PNG images')
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'pngs')
    
    convert_tiff_to_png(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
