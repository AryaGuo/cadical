#wget https://baldur.iti.kit.edu/sat-competition-2017/benchmarks/Main.zip
wget http://sat2018.forsyte.tuwien.ac.at/benchmarks/Main.zip
mkdir Main-18
unzip Main.zip -d Main-18
rm Main.zip
#find ./Main-18 -name "*.bz2" -exec bzip2 -d {} +
