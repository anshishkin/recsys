cd ./..

docker-compose \
    -f docker-compose.yml \
    -f dev.docker-compose.yml \
    run --rm kernel_svd