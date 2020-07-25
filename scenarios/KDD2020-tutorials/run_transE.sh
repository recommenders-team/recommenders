
echo $PWD
git clone https://github.com/thunlp/Fast-TransX.git
cd Fast-TransX
cd transE
g++ transE.cpp -o transE -pthread -O3 -march=native

inpath="../../data_folder/my/KG/"
outpath="../../data_folder/my/KG/"
if [ ! -d $outpath ]; then
  mkdir -p $outpath;
fi

./transE -size 32 -sizeR 32 -input $inpath  -output  $outpath  -epochs 10 -alpha 0.001
