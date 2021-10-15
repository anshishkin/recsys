cd ./..
docker login -u "$CI_REGISTRY_USER" -p "$CI_REGISTRY_PASSWORD" $CI_REGISTRY
export COMPOSE_FILE=docker-compose.yml:.ci/compose/merge.docker-compose.yml
docker-compose run --rm -T kernel_svd