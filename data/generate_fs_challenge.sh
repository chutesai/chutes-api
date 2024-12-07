#!/bin/bash

set -eu

# Previous run cleanup.
rm -f /tmp/base_filelist /tmp/target_filelist /tmp/workdir_filelist /tmp/*.data

target_image=$1
base_image=$(grep -E '^\s*FROM' Dockerfile | head -n 1 | awk -F 'FROM' '{print $NF}' | sed 's/ //g')
if [ -z "$base_image" ]
then
  echo "No base image detected?"
  exit 1
fi
working_dir=$(buildah inspect $target_image | jq -r '.OCIv1.config.WorkingDir')

echo "base_image=$base_image target_image=$target_image working_dir=$working_dir"

# Mount the images.
base_container=$(buildah from $base_image)
target_container=$(buildah from $target_image)
base_mountpoint=$(buildah mount $base_container)
mountpoint=$(buildah mount $target_container)

# Get a complete file list for each.
find $base_mountpoint \( -path $mountpoint/dev -o -path $mountpoint/proc -o -path $mountpoint/tmp -o -path $mountpoint/boot -o -path $mountpoint/run \) -prune -o -type f -printf '/%P\n' | sort | grep -E -v 'site-packages|dist-packages|\.cache' > /tmp/base_filelist
find $mountpoint \( -path $mountpoint/dev -o -path $mountpoint/proc -o -path $mountpoint/tmp -o -path $mountpoint/boot -o -path $mountpoint/run \) -prune -o -type f -printf '/%P\n' | sort | grep -E -v 'site-packages|dist-packages|\.cache' > /tmp/target_filelist
comm -13 /tmp/base_filelist /tmp/target_filelist > /tmp/allnew
if [ -z "$working_dir" ]
then
  working_dir="/app"
fi
find $mountpoint$working_dir \( -path $mountpoint/dev -o -path $mountpoint/proc -o -path $mountpoint/tmp -o -path $mountpoint/boot -o -path $mountpoint/run \) -prune -o -type f -printf "$working_dir/%P\n" | sort | grep -E -v 'site-packages|dist-packages|\.cache' > /tmp/workdir_filelist
file_size=$(stat -c%s -- "/tmp/workdir_filelist")
if [ "$file_size" -gt 0 ]
then
  comm -13 /tmp/base_filelist /tmp/workdir_filelist > /tmp/workdir
  comm -13 /tmp/workdir /tmp/allnew > /tmp/a && mv -f /tmp/a /tmp/allnew
fi
cat /tmp/allnew | shuf | head -n 1000 > /tmp/newsample

# Core libs (graval/chutes)
find $mountpoint \( -path $mountpoint/dev -o -path $mountpoint/proc -o -path $mountpoint/tmp -o -path $mountpoint/boot -o -path $mountpoint/run \) -prune -o -type f -printf '/%P\n' | sort | grep -E 'packages/(chutes|graval)/' > /tmp/corelibs

# Grab some data from each file.
for input_path in /tmp/newsample /tmp/workdir /tmp/corelibs
do
  if [ ! -f "$input_path" ]
  then
    continue
  fi
  data_path=/tmp/fschallenge_$(echo $input_path | awk -F '/' '{print $NF}').data
  rm -f $data_path
  echo ""
  echo $input_path | (grep -q newsample && echo "Generating filesystem challenge data for root filesystem...") || echo "Generating filesystem challenge data for working and app directories..."
  echo "=========================================="
  while IFS= read -r trailing_file_path || [ -n "$trailing_file_path" ]
  do
    file_path=$mountpoint$trailing_file_path
    file_path="${file_path#"${file_path%%[![:space:]]*}"}"
    file_path="${file_path%"${file_path##*[![:space:]]}"}"
    file_size=$(stat -c%s -- "$file_path")
    if [ "$file_size" -gt 1 ]
    then
      head -c 8192 -- "$file_path" > /tmp/head.data
      head_data=$(cat /tmp/head.data | base64 --wrap=0)
      tail_data="NONE"
      if [ "$file_size" -gt 8192 ]
      then
        tail -c 8192 "$file_path" > /tmp/tail.data
        tail_data=$(cat /tmp/tail.data | base64 --wrap=0)
      fi
      sha256_=$(sha256sum -- "$file_path" | awk '{print $1}')
      printf '%s:__size__:%s:__checksum__:%s:__head__:%s:__tail__:%s\n' "$trailing_file_path" "$file_size" "$sha256_" "$head_data" "$tail_data" >> $data_path
      printf 'Generated challenge data: %s\n' "$trailing_file_path"
    fi
  done < "$input_path"
done
