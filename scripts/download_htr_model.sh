set -e
wget 'https://sharedocs.huma-num.fr/wl/?id=0XRsGgjBwqPd0JLIwxRqW3w154PeURhi&fmode=download' --output-document trained_networks.zip
unzip trained_networks.zip && rm trained_networks.zip
mv trained_networks HTR/