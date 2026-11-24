#ifndef GUFO_EXT_GSQ_IDENTITY_HPP_
#define GUFO_EXT_GSQ_IDENTITY_HPP_

// Stable artifact fingerprint for flavor containers: region sizes, each
// region's head/tail bytes and the full tensor table (names, shapes,
// types, offsets). Purpose is runner/cache identity only; the scheme is
// extension-owned, exactly as in the standalone reader it replaces.

#include <string>

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::ext::gsq {

[[nodiscard]] std::string GsqIdentityHex(const core::GgufReader& reader);

}  // namespace gufo::ext::gsq

#endif  // GUFO_EXT_GSQ_IDENTITY_HPP_
