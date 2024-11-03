#!/bin/bash

set -eux

buildah containers --quiet | while read container_id
do
  buildah unmount $container_id --force
  buildah rm $container_id --force
done

buildah images --quiet | while read image_id
do
  buildah rmi $image_id --force
done
