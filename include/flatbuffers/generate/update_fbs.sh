set -e

pushd ./
./include/flatbuffers/generate/flatc --python -o ./pychatty/packer/fbs --gen-object-api ./include/flatbuffers/generate/model.fbs
./include/flatbuffers/generate/flatc --cpp --reflect-names -o ./include/flatbuffers ./include/flatbuffers/generate/model.fbs
popd
