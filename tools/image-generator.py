import argparse
import glob
import io
import os

from PIL import Image, ImageFont, ImageDraw

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/english.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../train_fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../english_images')

# Width and height of the resulting image.
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'train')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    total_count = 0
    prev_count = 0
    font_count = 0
    char_no = 0

    # Total number of font files is 
    print('total number of fonts are ', len(fonts))

    for character in labels:
        char_no += 1
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            total_count += 1
            font_count += 1
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color="white")
            font = ImageFont.truetype(font, 90)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                character,
                fill="black",
                font=font
            )
            file_string = '{}_{}.png'.format(font_count,char_no)
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'PNG')
        font_count = 0
    char_no = 0

    print('Finished generating {} images.'.format(total_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)
