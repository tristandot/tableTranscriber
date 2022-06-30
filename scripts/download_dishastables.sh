set -e
wget 'https://sharedocs.huma-num.fr/wl/?id=PeJAYwb6KdfBh9d3S9qeCwq6xiduAUUi&fmode=download' --output-document DishasTables.zip
unzip DishasTables.zip && rm DishasTables.zip
mv DishasTables tableExtractor/datasets/
