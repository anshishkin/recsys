cd ./..

docker-compose \
    -f docker-compose.yml \
    -f .ci/compose/prod.docker-compose.yml \
    up