import argparse
import itertools
from pathlib import Path

import cv2
import huffman
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


# Step 1: Convert image to YCbCr space
def rgb2ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# Step 1.2: Pad the image so its dimensions are multiples of 8


def pad_image(image):
    h, w = image.shape[:2]

    # block_size = 8
    new_h = h if h % 8 == 0 else (h // 8 + 1) * 8
    new_w = w if w % 8 == 0 else (w // 8 + 1) * 8

    padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded_image[:h, :w, :] = image

    return padded_image

# Step 2: Divide the input image into 8x8 blocks


def divide_into_blocks(image):
    h, w = image.shape[:2]
    blocks = []

    # Iterate over the image in steps of block_size to extract blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8]
            if block.shape == (8, 8, 3):
                blocks.append(block)
    return blocks

# Step 3: Perform Discrete Cosine Transform (DCT) on each 8x8 block


def apply_dct(block):
    # DCT transform along rows and then along columns
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Step 4: Apply quantization table to each 8x8 block


def quantize(block, q_table_y, q_table_cbcr):
    # quantized block of the same shape as the input block
    quantized_block = np.zeros_like(block)
    quantized_block[:, :, 0] = (
        block[:, :, 0] / q_table_y).round().astype(int)  # Y channel
    quantized_block[:, :, 1] = (
        block[:, :, 1] / q_table_cbcr).round().astype(int)  # Cb channel
    quantized_block[:, :, 2] = (
        block[:, :, 2] / q_table_cbcr).round().astype(int)  # Cr channel
    return quantized_block

# Step 5: Apply zig-zag ordering


def zigzag_order(block):
    indices = [(x, y) for x in range(8)
               for y in range(8)]  # list of indices for zig-zag order
    zigzag_indices = sorted(indices, key=lambda xy: (
        xy[0] + xy[1], xy[0] if (xy[0] + xy[1]) % 2 == 0 else -xy[0]))
    return np.array([block[x, y] for x, y in zigzag_indices])

# Step 6: Apply run-length encoding


def run_length_encoding(arr):
    rle = []

    # Store the first element and start count at 1
    prev = tuple(arr[0])
    count = 1

    # Iterate through the array to encode runs of identical values
    for current in arr[1:]:
        current_tuple = tuple(current)
        if np.array_equal(current_tuple, prev):
            count += 1
        else:
            rle.append((prev, count))
            prev = current_tuple
            count = 1
    # Append the final run to the list
    rle.append((prev, count))

    return rle

# Step 7: Apply Huffman encoding


def huffman_encoding(rle):
    # Flatten run-length encoded data for Huffman encoding
    flattened = [tuple(symbol) for symbol, count in rle for _ in range(count)]

    # Calculate frequency of each symbol
    frequencies = {symbol: flattened.count(
        symbol) for symbol in set(flattened)}

    # Create Huffman codebook from frequencies
    huffman_tree = huffman.codebook(frequencies.items())

    # Encode the data using the Huffman codebook
    encoded_data = ''.join(huffman_tree[symbol] for symbol in flattened)

    return encoded_data, huffman_tree

# Decompress image


def huffman_decoding(encoded_data, codebook):
    # Reverse the Huffman codebook to decode
    reverse_codebook = {v: k for k, v in codebook.items()}
    current_code = ""
    decoded_data = []

    # Decode Huffman encoded data
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codebook:
            decoded_data.append(reverse_codebook[current_code])
            current_code = ""

    return decoded_data


def inverse_zigzag_order(arr):
    indices = [(x, y) for x in range(8)
               for y in range(8)]  # indices for inverse zig-zag ordering
    zigzag_indices = sorted(indices, key=lambda xy: (
        xy[0] + xy[1], xy[0] if (xy[0] + xy[1]) % 2 == 0 else -xy[0]))
    block = np.zeros((8, 8, 3), dtype=int)
    for i, (x, y) in enumerate(zigzag_indices):
        block[x, y] = arr[i]
    return block


def dequantize(block, q_table_y, q_table_cbcr):
    dequantized_block = np.zeros_like(block)
    dequantized_block[:, :, 0] = block[:, :, 0] * q_table_y  # Y channel
    dequantized_block[:, :, 1] = block[:, :, 1] * q_table_cbcr  # Cb channel
    dequantized_block[:, :, 2] = block[:, :, 2] * q_table_cbcr  # Cr channel
    return dequantized_block


def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def reconstruct_image(blocks, h, w):
    # Initialize an empty image
    image = np.zeros((h, w, 3))
    block_idx = 0

    # Place each block back into its original position in the image
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # print(f"(i,j)=({i},{j}), j+8={j+8}, image.shape: {image[i:i+8, j:j+8].shape}, blocks.shape: {blocks[block_idx].shape}")
            image[i:i+8, j:j+8] = blocks[block_idx]
            block_idx += 1

    # Ensure pixel values are within valid range
    return np.clip(image, 0, 255).astype(np.uint8)


def get_quantization_tables(quality):
    # Taken from https://www.sciencedirect.com/topics/engineering/quantization-table

    # Standard JPEG quantization table for quality 50
    luminance_q_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                 [12, 12, 14, 19, 26, 58, 60, 55],
                                 [14, 13, 16, 24, 40, 57, 69, 56],
                                 [14, 17, 22, 29, 51, 87, 80, 62],
                                 [18, 22, 37, 56, 68, 109, 103, 77],
                                 [24, 35, 55, 64, 81, 104, 113, 92],
                                 [49, 64, 78, 87, 103, 121, 120, 101],
                                 [72, 92, 95, 98, 112, 100, 103, 99]])

    chrominance_q_table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                    [18, 21, 26, 66, 99, 99, 99, 99],
                                    [24, 26, 56, 99, 99, 99, 99, 99],
                                    [47, 66, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99]])

    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - 2 * quality

    # Adjust the quantization tables based on the scale_factor
    q_table_y = np.floor(
        (luminance_q_table * scale_factor + 50) / 100).astype(np.int32)
    q_table_cbcr = np.floor(
        (chrominance_q_table * scale_factor + 50) / 100).astype(np.int32)

    # Ensure no zero values in quantization tables
    q_table_y[q_table_y == 0] = 1
    q_table_cbcr[q_table_cbcr == 0] = 1

    return q_table_y, q_table_cbcr


def main(image_path, quality=100):
    # 1. RGB -> YCbCr
    input_image = np.array(Image.open(image_path))
    h, w = input_image.shape[:2]
    # Pad image to be a multiple of 8 on height and width
    ycbcr_image = pad_image(rgb2ycbcr(input_image))

    # 2. 8x8 blocks
    blocks = divide_into_blocks(ycbcr_image)

    # higher - better quality, lower - more compression
    q_table_y, q_table_cbcr = get_quantization_tables(quality)

    # 3. Compression
    compressed_blocks, codebooks = compress(blocks, q_table_y, q_table_cbcr)

    # 4. Decompression
    decompressed_blocks = decompress(
        compressed_blocks, codebooks, q_table_y, q_table_cbcr)

    # 5. Image reconstruction from decompressed blocks
    reconstructed_image = reconstruct_image(
        decompressed_blocks, ycbcr_image.shape[0], ycbcr_image.shape[1])
    reconstructed_image = reconstructed_image[:h, :w, :]    # Crop image
    reconstructed_image_rgb = cv2.cvtColor(
        reconstructed_image, cv2.COLOR_YCrCb2RGB)

    output_image = Image.fromarray(reconstructed_image_rgb)
    return output_image


def compress(blocks, q_table_y, q_table_cbcr):
    compressed_blocks = []
    codebooks = []

    # Compress
    for block in blocks:
        dct_block = apply_dct(block)
        quantized_block = quantize(dct_block, q_table_y, q_table_cbcr)
        zigzagged_block = zigzag_order(quantized_block)
        rle = run_length_encoding(zigzagged_block)
        huffman_encoded, codebook = huffman_encoding(rle)
        compressed_blocks.append(huffman_encoded)
        codebooks.append(codebook)

    return compressed_blocks, codebooks


def decompress(compressed_blocks, codebooks, q_table_y, q_table_cbcr):
    decompressed_blocks = []

    # Decompress
    for encoded_data, codebook in zip(compressed_blocks, codebooks):
        decoded_data = huffman_decoding(encoded_data, codebook)
        rle = [(k, len(list(g))) for k, g in itertools.groupby(decoded_data)]
        zigzagged_block = [symbol for symbol,
                           count in rle for _ in range(count)]
        zigzagged_block = np.array(zigzagged_block)
        quantized_block = inverse_zigzag_order(zigzagged_block)
        dequantized_block = dequantize(
            quantized_block, q_table_y, q_table_cbcr)
        idct_block = apply_idct(dequantized_block)
        decompressed_blocks.append(idct_block)

    return decompressed_blocks


if __name__ == '__main__':
    image_path = './flower.png'  # args.image_path
    quality = 50    # args.image_path   # Adjust this value to control the compression quality

    # Lossless-JPEG main algorithm
    print("[INFO] JPEG compression started...")
    output_image = main(image_path, quality)
    print("[+] JPEG compression ended successfully!")

    outName = '{}(quality={}).jpg'.format(Path(image_path).stem, quality)
    output_image.save(outName)
    output_image.show()

    print(
        f"JPEG compression and decompression applied. Compressed image saved as '{outName}'.")
