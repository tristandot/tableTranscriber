set -e
wget 'https://sharedocs.huma-num.fr/wl/?id=LwY4cwePM8KnET4MaUGSYRTyMlVPatlu&fmode=download' --output-document tables_cells.zip
unzip tables_cells.zip && rm tables_cells.zip
mkdir -p HTR/datasets
mv tables_cells HTR/datasets/