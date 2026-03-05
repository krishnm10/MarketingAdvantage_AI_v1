sudo dnf update -y

# Install Docker engine
sudo dnf install -y docker

# Start and enable service
sudo systemctl enable docker
sudo systemctl start docker

# Let your user run docker
sudo usermod -aG docker ec2-user   # or your username
# then log out and log back in

mkdir -p $HOME/qdrant_storage

docker pull qdrant/qdrant:latest

docker run -d \
  --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$HOME/qdrant_storage:/qdrant/storage" \
  -e QDRANT__SERVICE__API_KEY='my-super-secret-admin-key' \
  qdrant/qdrant:latest

curl http://localhost:6333/healthz
