for i in {01..10}; do
  tar -xzf data/v1.0-trainval${i}_blobs.tgz -C ./data/nuscenes/
done