export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd ./models/GroundingDINO/ops
python3 setup.py build
pip3 install .
python ./test.py # should result in 6 lines of * True
cd ../../../
