#!/bin/bash
#Change here to download a specific Height and Pitch Variation, for example - H_15_P_0
# HP=('H_15_P_45' 'H_15_P_60' 'H_15_P_90')
HP=('H_15_P_90')

#Change here to download a specific weather subset, for example - ClearNoon
#Note - For Segment, Instance and Depth annotations this field should only have ClearNoon variation
# weather=('ClearNoon' 'ClearNight')
weather=('ClearNoon')

#Change here to download a specific Town subset, for example - Town07
layout=('Town07')

#Change here for any specific annotation, for example - https://huggingface.co/datasets/hoffman-lab/SkyScenes/resolve/main/Segment
base_url=('https://huggingface.co/datasets/hoffman-lab/SkyScenes/resolve/main/Images')

#Change here for base download folder
base_download_folder='SkyScenes'


for hp in "${HP[@]}"; do
  for w in "${weather[@]}"; do
      for t in "${layout[@]}"; do
        folder=$(echo "$base_url" | awk -F '/' '{print $(NF)}')
        download_url="${base_url}/${hp}/${w}/${t}/${t}.tar.gz"
        download_folder="${base_download_folder}/${folder}/${hp}/${w}/${t}"
        mkdir -p "$download_folder"
        echo "Downloading: $download_url"
        wget -P "$download_folder" "$download_url"
      done
  done
done
