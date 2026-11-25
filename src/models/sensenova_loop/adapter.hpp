// src/models/sensenova_loop/adapter.hpp - the sensenova_loop (SenseNova Loop: a looped qwen35 dense hybrid with two-stream
// hyper-connections) adapter.
#pragma once

#include "basalt/model/adapter.hpp"

#include <memory>

namespace basalt::models::sensenova_loop {
std::unique_ptr<model::ModelAdapter> make_adapter();
}  // namespace basalt::models::sensenova_loop
