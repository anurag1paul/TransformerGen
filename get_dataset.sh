wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz CUB_200_2011.tgz CUB_200_2011/images
tar -xvf CUB_200_2011.tgz CUB_200_2011/images.txt
curl -L "https://drive.google.com/uc?export=download&id=1HXnzREyrcYkJiF00rIsyjgA6X3ljPK40" > labels.tar.gz
tar -xvf labels.tar.gz
mkdir dataset 
mv images dataset
mv images.txt dataset
mv text_c10 dataset
rm -rf CUB_200_2011
