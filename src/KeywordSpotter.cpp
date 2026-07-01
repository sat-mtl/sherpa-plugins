// The implementation lives entirely in KeywordSpotter.hpp (inline): several
// Avendish back-ends compile the header directly and do not link this library, so
// a .cpp-defined operator() would be an unresolved external. This TU only exists
// to give the CMake target a source.
#include "KeywordSpotter.hpp"
