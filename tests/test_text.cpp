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
  const std::string input = "how are you doing today it is a nice day";
  obj.inputs.text.value = input;
  obj();

  const std::string out = obj.outputs.result.value;
  INFO("input:  " << input);
  INFO("output: " << out);
  REQUIRE_FALSE(out.empty());

  // The processor must actually TRANSFORM the text, not echo it. A punctuation
  // model restores punctuation and/or casing; a diacritization model adds marks.
  // Either way the output must differ from the bare lowercase input, and it must
  // preserve the content (never shorter than the input).
  CHECK(out != input);
  CHECK(out.size() >= input.size());

  const bool has_punct = out.find_first_of(".,?!;:،؟") != std::string::npos;
  bool has_upper = false;
  bool has_nonascii = false; // diacritics (Arabic tashkeel etc.) are multi-byte
  for(unsigned char c : out)
  {
    if(c >= 'A' && c <= 'Z')
      has_upper = true;
    if(c >= 0x80)
      has_nonascii = true;
  }
  INFO("has_punct=" << has_punct << " has_upper=" << has_upper
                    << " has_nonascii=" << has_nonascii);
  // A meaningful transformation adds punctuation, casing, or diacritics.
  CHECK((has_punct || has_upper || has_nonascii));
}
