// src/models/sensenova_loop/meta_test.cpp - a sensenova_loop pack's metadata into the dense geometry: the visits' plan (the
// prelude once, the loop's layers count times, the coda once), each visit's kind, the hyper-connections, the YaRN keys.
// The pack: the first argument or BASALT_SENSENOVA_LOOP_PACK (skipped without one).
#include "models/sensenova_loop/meta.hpp"

#include "basalt/core/model_file.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace sl = basalt::models::sensenova_loop;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("BASALT_SENSENOVA_LOOP_PACK");
    if (!path) {
        std::printf("no pack (argument or BASALT_SENSENOVA_LOOP_PACK): skipped\n");
        return 77;
    }
    int failures = 0;
    const auto check = [&](bool c, const char* what) {
        if (!c) {
            std::printf("FAIL %s\n", what);
            ++failures;
        }
    };
    basalt::core::ModelFile mf;   // the pack's metadata as a GGUF view, as the engine opens it
    std::string open_err;
    if (!mf.open(path, open_err) || !mf.load_meta(open_err) || !mf.register_gguf_view(open_err)) {
        std::printf("FAIL %s: %s\n", path, open_err.c_str());
        return 1;
    }
    const basalt::GgufFile gg(path);
    check(sl::validate_meta(gg).empty(), "validate_meta");
    sl::Loop loop;
    std::string err;
    const basalt::dense::Geometry g = sl::geometry(gg, loop, err);
    check(err.empty(), err.empty() ? "geometry" : err.c_str());
    const int stored = (int) loop.stored.size();
    const int width = loop.end - loop.start;
    check(g.n_layers == stored + (loop.count - 1) * width, "visits = stored + (count - 1) x the loop's layers");
    check(g.stored_layers() == stored, "stored_layers");
    for (int il = 0; il < g.n_layers; ++il) {
        int want = il;
        if (il >= loop.start + loop.count * width) want = il - (loop.count - 1) * width;
        else if (il >= loop.start) want = loop.start + (il - loop.start) % width;
        check(g.layer_of(il) == want, "the plan");
        check(g.kinds[(size_t) il] == loop.stored[(size_t) want], "each visit's kind");
    }
    check(g.hc_streams == 2 && g.hc_iters >= 1, "two-stream hyper-connections");
    check(loop.hc_start >= loop.start && loop.hc_end <= loop.end, "the hyper-connections inside the loop");
    const basalt::MetaValue* yarn = sl::key(gg, "rope.scaling.type");
    check(yarn && yarn->s == "yarn", "YaRN keys");
    std::printf("%d stored layers, %d visits (%d attention, %d GDN), loop %d-%d x%d, hyper-connections %d-%d, %d Sinkhorn rounds\n",
                stored, g.n_layers, g.attn_layers(), g.gdn_layers(), loop.start, loop.end - 1, loop.count, loop.hc_start,
                loop.hc_end - 1, g.hc_iters);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
