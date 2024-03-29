#!/bin/bash

# Base directory
base_dir="/workspaces/text-web-and-media-analytics-assignment-2"

# Execute poetry update in specific directories
cd "$base_dir/python" && poetry update &

# Wait for all background jobs to complete
wait