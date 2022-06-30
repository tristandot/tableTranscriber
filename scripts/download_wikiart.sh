set -e
wget 'http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip' --output-document wikiart.zip
unzip wikiart.zip && rm wikiart.zip
mkdir -p tableExtractor/synthetic_resource
mv wikiart tableExtractor/synthetic_resource/
