#pragma once
// RT-safe mono accumulation for the audio-driven objects, plus notes on the
// threading contract.
//
// Threading contract (see also the avendish-rt-safety memory / README):
//   * operator() runs on the audio/DSP thread: NO allocation, NO mutex, NO
//     inference. It only downmixes the incoming block into a pre-reserved buffer
//     (MonoAccum::push) and, when an inference is not already in flight, hands the
//     accumulated audio to the Avendish `worker` (a copy into a reused job + a
//     shared_ptr bump -- allocation-free in steady state).
//   * worker::work() runs on a pool thread: it (re)loads the model and runs
//     sherpa-onnx inference -- slow work is fine here.
//   * The std::function<void(Self&)> that work() returns is applied back on the
//     processing thread by the host; it only moves/swaps results into output
//     ports and clears the in-flight flag -- still allocation-free.

#include <cstddef>
#include <vector>

namespace sherpa
{

// Accumulates a mono downmix of successive audio blocks into a pre-reserved
// buffer. Overflow beyond the reserved capacity is dropped (logged by the caller
// if desired) rather than reallocating on the audio thread.
struct MonoAccum
{
  std::vector<float> buf;

  void prepare(std::size_t capacity_samples)
  {
    buf.clear();
    buf.reserve(capacity_samples);
  }

  // Bus is a halp audio bus: bus[c] yields channel c's float* for this block.
  template <typename Bus>
  void push(const Bus& bus, int channels, int frames) noexcept
  {
    if(channels <= 0 || frames <= 0)
      return;
    if(buf.size() + static_cast<std::size_t>(frames) > buf.capacity())
      return; // RT cap reached: drop rather than reallocate
    const float inv = channels > 1 ? 1.f / static_cast<float>(channels) : 1.f;
    for(int i = 0; i < frames; ++i)
    {
      float s = 0.f;
      for(int c = 0; c < channels; ++c)
        s += bus[c][i];
      buf.push_back(s * inv);
    }
  }

  bool empty() const noexcept { return buf.empty(); }
  std::size_t size() const noexcept { return buf.size(); }

  // Move the accumulated samples into `out` (reusing out's capacity) and clear
  // the accumulator, keeping its own capacity. Both stay allocation-free once
  // their capacities are warm.
  void drain_into(std::vector<float>& out)
  {
    out.assign(buf.begin(), buf.end());
    buf.clear();
  }
};

}
