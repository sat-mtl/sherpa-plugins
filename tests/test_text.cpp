// Workflow test for sherpa_text: punctuation restoration / diacritization.
// Requires: SHERPA_TEST_TEXT_MODEL = a punctuation or diacritization model dir.
#include "TestHelpers.hpp"

#include "TextProcessor.hpp"

#include <catch2/catch_all.hpp>

using namespace sherpa;

TEST_CASE("Text processor transforms input text", "[text][workflow]")
{
  if(!test::loader_ready())
    SHERPA_SKIP("sherpa-onnx runtime library not available");
  auto model = test::env("SHERPA_TEST_TEXT_MODEL");
  if(!model)
    SHERPA_SKIP("set SHERPA_TEST_TEXT_MODEL");

  TextProcessor obj;
  test::wire_sync_worker(obj);
  obj.inputs.model.value = *model;
  obj.inputs.threads.value = 1;
  // Feeding text triggers on_input() -> operator() -> (sync) worker.
  obj.inputs.text.value = "how are you doing today it is a nice day";
  obj();

  INFO("output: " << obj.outputs.result.value);
  REQUIRE_FALSE(obj.outputs.result.value.empty());
}
