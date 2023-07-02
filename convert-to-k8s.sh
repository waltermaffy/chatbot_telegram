echo "Converting docker-compose to k8s"
docker-compose config > docker-compose-resolved.yaml && kompose convert -f docker-compose-resolved.yaml --out  ./k8s
echo "Done"