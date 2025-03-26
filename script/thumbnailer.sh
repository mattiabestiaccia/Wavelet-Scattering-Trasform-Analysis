#!/bin/bash

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

filename=${1:-"/home/brus/Projects/wavelet/images/ortho/190529_pi_5ab_re/canale_0.tif"}
output=${1:-"/home/brus/Projects/wavelet/images/ortho/190529_pi_5ab_re/canale_0_thumb.jpg"}

# Crea una miniatura dell'immagine specificata
vips thumbnail "$filename" "$output" 200

# Visualizza l'immagine
if command -v eog &> /dev/null
then
	eog "$output" &
else
	apt update && apt install -y eog
	eog "$output" &

fi