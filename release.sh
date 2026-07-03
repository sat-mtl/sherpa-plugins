#!/bin/bash
# Source package for the score package manager (mirrors score-addon-cv /
# score-addon-onnx). Run by the ossia/actions release track on every push; the
# resulting zip is uploaded to the GitHub release on v* tags (the upload glob
# there is score-addon-*.zip, hence the folder name). Ships everything needed
# to compile the addon: sources, the CMake build, and the vendored sherpa-onnx
# C header (the runtime library is dlopen'd, never linked).
rm -rf release
mkdir -p release

cp -rf src cmake 3rdparty CMakeLists.txt addon.json README.md release/
[ -f LICENSE ] && cp -f LICENSE release/

mv release score-addon-sherpa
7z a score-addon-sherpa.zip score-addon-sherpa
