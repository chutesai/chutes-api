#!/bin/bash

set -eu

target_image=$1

# Mount the image.
target_container=$(buildah from $target_image)
mountpoint=$(buildah mount $target_container)

# Scan.
trivy fs --severity HIGH,CRITICAL --scanners vuln --ignore-unfixed $mountpoint
