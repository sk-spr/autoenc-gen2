#!/bin/bash

echo "downloading..."
./download_at_dtm.sh
echo "partitioning..."
./create_tiles.sh
echo "Done."
