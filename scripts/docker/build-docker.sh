# build the CPU and GPU docker images

git archive --format=tar -o docker/memory.tar --prefix=memory/ master
docker build --no-cache -t memory/gpu -f docker/gpu/Dockerfile .
docker build --no-cache -t memory/cpu -f docker/cpu/Dockerfile .
rm docker/memory.tar
