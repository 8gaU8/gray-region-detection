#! /bin/bash
set -euxoC pipefail
cd "$(dirname "$0")"

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

mkdir dataset
wget http://www.cs.sfu.ca/%7Ecolour/data2/shi_gehler/png_canon1d.zip -P dataset

wget http://www.cs.sfu.ca/%7Ecolour/data2/shi_gehler/png_canon5d_1.zip -P dataset
wget http://www.cs.sfu.ca/%7Ecolour/data2/shi_gehler/png_canon5d_2.zip -P dataset
wget http://www.cs.sfu.ca/%7Ecolour/data2/shi_gehler/png_canon5d_3.zip -P dataset

cd dataset
unzip png_canon1d.zip
unzip png_canon5d_1.zip
unzip png_canon5d_2.zip
unzip png_canon5d_3.zip

