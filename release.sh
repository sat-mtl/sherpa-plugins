#!/bin/bash
# Source package for the score package manager (mirrors score-addon-cv /
# score-addon-onnx). Run by the ossia/actions release track on every push; the
# resulting zip is uploaded to the GitHub release on v* tags (the upload glob
# there is score-addon-*.zip, hence the folder name). Ships everything needed
# to compile the addon: sources, the CMake build, and the vendored sherpa-onnx
# C header (the runtime library is dlopen'd, never linked).
rm -rf release
mkdir -p release

# phonemizer/ too: its header is part of the build (phonemizer/include).
cp -rf src cmake 3rdparty phonemizer CMakeLists.txt addon.json README.md release/
[ -f LICENSE ] && cp -f LICENSE release/

# A package made from a v* tag downloads the prebuilt sherpa-onnx archives
# attached to that same release (.github/workflows/sherpa-prebuilt.yml), falling
# back to the version-pair release while they are still building. Any other
# package uses the version-pair release (cmake/sherpa-onnx.cmake).
if [[ "${GITHUB_REF_TYPE:-}" == "tag" && "${GITHUB_REF_NAME:-}" == v* ]]; then
  echo "${GITHUB_REF_NAME}" > release/cmake/sherpa-onnx-prebuilt-tag.txt
fi

mv release score-addon-sherpa
7z a score-addon-sherpa.zip score-addon-sherpa
