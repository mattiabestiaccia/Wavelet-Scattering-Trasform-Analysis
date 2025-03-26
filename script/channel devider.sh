#!/bin/bash

# Cartella di output
output_dir="/home/brus/Projects/wavelet/images/ortho/190529_pi_5ab_re"
# mkdir -p "$output_dir"

# File TIFF di input
input_file="/home/brus/Projects/wavelet/images/ortho/190529_pi_5ab_re.tif"
num_bands=5

# Estrai ogni canale e salvalo
for ((band=0; band<num_bands; band++)); do
    output_file="${output_dir}/canale_${band}.tif"
    vips extract_band "$input_file" "$output_file" $band
    echo "Canale $band salvato in $output_file"
    echo "Progresso: $((band + 1))/$num_bands"
done

echo "Estrazione completata. I canali sono stati salvati nella cartella '$output_dir'."