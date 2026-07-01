#pragma once
// Small shared bits used across objects: the execution-provider selector.

namespace sherpa
{

// onnxruntime execution provider. sherpa falls back to CPU if the requested one
// is unavailable in the shipped onnxruntime, so exposing all of them is safe.
enum class Provider
{
  CPU,
  CUDA,
  CoreML,
  DirectML
};

inline const char* provider_str(Provider p) noexcept
{
  switch(p)
  {
    case Provider::CUDA:
      return "cuda";
    case Provider::CoreML:
      return "coreml";
    case Provider::DirectML:
      return "directml";
    case Provider::CPU:
    default:
      return "cpu";
  }
}

}
