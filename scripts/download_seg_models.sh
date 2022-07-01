set -e
wget 'https://sharedocs.huma-num.fr/wl/?id=U7vDeozr8nuqr9z23uSc9B4AY7gWhXyI&fmode=download' --output-document models.zip
unzip models.zip && rm models.zip && rm -r __MACOSX
mv models tableExtractor/
