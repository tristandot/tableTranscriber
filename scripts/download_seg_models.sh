set -e
wget 'https://sharedocs.huma-num.fr/wl/?id=lKSGOeOoXvjIDjlaRQqUdyx2wHXBBZri&fmode=download' --output-document models.zip
unzip models.zip && rm models.zip
mv models tableExtractor/
