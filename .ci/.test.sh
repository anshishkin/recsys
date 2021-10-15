cd ./..

docker-compose \
    -f docker-compose.yml \
    -f .ci/compose/test.docker-compose.yml \
    run --rm -T kernel_svd