// The extension entry. Providers register per provider (gsq/provider.cpp);
// the Flash-Next runtime comes from the qwen38_flash_next dependency.
#include <gufo/ext/registry.hpp>

void qwen38_flash_next_flavors_register(gufo::ext::Registry& registry) {
  (void)registry;
}
