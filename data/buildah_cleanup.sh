#!/bin/bash

set -eux

buildah containers --quiet | while read container_id
do
  buildah unmount $container_id
  buildah rm $container_id
done

buildah images --quiet --filter "reference!=docker.io/parachutes/*" | while read image_id
do
  buildah rmi --force $image_id
done
