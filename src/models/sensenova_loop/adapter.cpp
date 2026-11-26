// src/models/sensenova_loop/adapter.cpp - SenseNova Loop (SenseNova-Loop-9B): registers the architecture, reads its pack into
// the shared dense stack (its loop as the visits' plan, its hyper-connection routers on their layers) and hands out its
// sessions.  One card: the loop's layers run twice, so a layer cut or a prompt stage would hold some layers on both cards.
#include "models/sensenova_loop/adapter.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/dense/pack.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/model/rope_flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/qwen_forms.hpp"
#include "models/sensenova_loop/meta.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>

namespace basalt::models::sensenova_loop {
namespace {

constexpr const char* kArea = "basalt sensenova_loop";

// The dense stack's options for a run of these flags (when drafting, the pack's DFlash 2 block drafter, else its MTP layer).
dense::Options run_options(const GgufFile& gg, const model::LoadRequest& req) {
    bool mtp = false, block = false;
    for (const TensorInfo& t : gg.tensors()) {
        mtp |= t.name == "mtp.fc.weight";
        block |= t.name == "dflash.fc.0.weight";
    }
    dense::Options o;
    o.max_context = req.max_context;
    o.sessions = req.sessions;
    o.spec = req.spec;
    o.draft = req.draft && (mtp || block);
    o.drafter = !o.draft ? dense::Drafter::None : block ? dense::Drafter::Block : dense::Drafter::Mtp;
    o.vision = req.vision;
    o.kv_int8 = req.kv == "int8" || req.kv == "nvfp4";
    o.kv_fp4 = req.kv == "nvfp4";
    o.kv_resident = req.kv_resident;
    o.arch = kArch;
    return o;
}

class Loaded final : public model::LoadedModel {
public:
    ~Loaded() override {
        model_.reset();
        reader_.reset();
    }

    const model::ModelDesc& desc() const override { return desc_; }

    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads, std::vector<std::unique_ptr<model::ModelSession>>&) override {
        std::string err;
        if (req.paths.empty()) return {model::Status::Error, "sensenova_loop: no model path"};
        if (req.gpus.cards.size() > 1)
            return {model::Status::Error, "sensenova_loop: one card only (the loop runs some layers twice; a second card would hold them too)"};
        if (!plan_load(req, reads, err)) return {model::Status::Error, err};
        return {};
    }

    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override {
        if (std::string err; !dense::attach_drafter(*model_, err)) return {model::Status::Error, err};
        for (int i = 0; i < std::max(req.sessions, 1); ++i) {
            std::string err;
            std::unique_ptr<model::ModelSession> s = dense::make_session(*model_, err);
            if (!s) return {model::Status::Error, err};
            sessions.push_back(std::move(s));
        }
        return {};
    }

private:
    bool plan_load(const model::LoadRequest& req, core::LoadPlan& reads, std::string& err) {
        const std::string& path = req.paths.front();
        const GgufFile gg(path);
        desc_ = describe_meta(gg);
        Loop loop;
        dense::Geometry g = geometry(gg, loop, err);
        if (!err.empty()) {
            err = "sensenova_loop: " + err;
            return false;
        }
        // the rope scaling of the run (rope_flags.hpp): the pack's (YaRN at every position, as the checkpoint is configured),
        // then the flags
        g.rope.scaling.freq_base = g.rope.theta;
        g.rope.scaling.orig_ctx = (double) (g.context_length > 0 ? g.context_length : 262144);
        if (!model::resolve_rope_scaling(req.rope, model::rope_keys(gg, kArch), req.max_context, kArea, g.rope.scaling, err)) {
            err = "sensenova_loop: " + err;
            return false;
        }
        g.rope.theta = (float) g.rope.scaling.freq_base;
        const int64_t longest = g.rope.scaling.type == kernels::RopeScalingType::None
                                    ? g.context_length
                                    : (int64_t) (g.rope.scaling.orig_ctx / g.rope.scaling.freq_scale());
        if (longest > 0 && req.max_context > longest) {
            err = "sensenova_loop: --max-context " + std::to_string((long long) req.max_context) + " is beyond the model's " +
                  std::to_string((long long) longest);
            return false;
        }
        dense::Options o = run_options(gg, req);
        const int file = plan_.file(path);
        reader_ = std::make_unique<dense::PackReader>(gg, "sensenova_loop", plan_, file);
        dense::PackReader& r = *reader_;
        const bool mtp = o.drafter == dense::Drafter::Mtp, block = o.drafter == dense::Drafter::Block;
        if (!r.reserve([&](const std::string& n) {
                return n.starts_with("blk.") || n.starts_with("output") || (mtp && n.starts_with("mtp.")) ||
                       (block && n.starts_with("dflash."));
            }, err))
            return false;
        dense::Weights w;
        const int H = g.n_embd;
        w.embd = r.host_embedding("token_embd.weight", g.n_vocab, H, err);   // the kernels read the rows they need
        if (!w.embd) return false;
        w.out_norm = r.floats("output_norm.weight", H, err);
        if (!w.out_norm || !r.linear("output", g.n_vocab, H, w.head, err)) return false;
        for (int il = 0; il < (int) loop.stored.size(); ++il) {   // the stored layers; the plan visits them
            dense::Layer L;
            if (!r.layer(g, il, loop.stored[(size_t) il], L, err)) return false;
            const bool hc = il >= loop.hc_start && il < loop.hc_end;
            if (hc != (L.hc.attn_w != nullptr)) {
                err = "sensenova_loop: layer " + std::to_string(il) + (hc ? " lacks its" : " has") + " hyper-connection routers";
                return false;
            }
            L.hc.enter = hc && il == loop.hc_start;
            L.hc.leave = hc && il == loop.hc_end - 1;
            w.layers.push_back(L);
        }
        if (mtp && !r.mtp(g, w.mtp, err)) return false;
        if (block && !r.block_drafter(g, kArch, w.dflash, err)) return false;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        basalt::log::info(kArea, "%zu tensors, %.2f GiB on the GPU, %.2f GiB of embedding in page-locked RAM; %.2f GiB free; "
                          "%d stored layers, %d visits (layers %d-%d run %d times), hyper-connections over layers %d-%d%s",
                          r.count(), (double) r.device_bytes() / 1073741824.0, (double) r.host_bytes() / 1073741824.0,
                          (double) free_b / 1073741824.0, (int) loop.stored.size(), g.n_layers, loop.start, loop.end - 1, loop.count,
                          loop.hc_start, loop.hc_end - 1,
                          block ? "; drafting with the DFlash 2 block drafter" : mtp ? "; drafting with the MTP layer" : "");
        o.host_embedding_bytes = r.host_bytes();
        o.gpu_weight_bytes = r.device_bytes();
        if (o.vision) {
            core::ModelFile mf;
            core::ModelSources srcs;
            if (!mf.open(path, err)) return false;
            core::model_sources(mf, srcs);
            o.mmproj = srcs.vision;
        }
        const uint64_t need_b = dense::shared_bytes(g, o.max_context, o.kv_format(), o.kv_resident) +
                                (uint64_t) o.sessions * (dense::session_bytes(g, o.max_context, mtp, o.kv_format(), o.kv_resident) +
                                                         (block ? dense::block_session_bytes(g, w.dflash, o.max_context) : 0));
        if (const uint64_t host_b = dense::session_host_bytes(g, o.max_context, o.kv_format(), o.kv_resident))
            basalt::log::info(kArea, "K/V streaming: %lld of %lld cells per attention visit in VRAM, %.2f GiB in page-locked RAM per slot",
                              (long long) dense::resident_cells(o.max_context, o.kv_resident), (long long) o.max_context,
                              (double) host_b / 1073741824.0);
        const uint64_t reserve = (uint64_t) std::max(req.vram_reserve_mib, 0) << 20;
        if (need_b + reserve + (512ull << 20) > free_b) {
            err = "sensenova_loop: " + std::to_string((long long) o.sessions) + " session(s) of " + std::to_string((long long) o.max_context) +
                  " cells need " + std::to_string((unsigned long long) (need_b >> 20)) + " MiB, " +
                  std::to_string((unsigned long long) (free_b >> 20)) + " MiB are free; lower --max-context or --parallel";
            return false;
        }
        model_ = dense::make_model(g, w, o, err);
        if (!model_) return false;
        reads.append(plan_);
        return true;
    }

    model::ModelDesc desc_;
    core::LoadPlan plan_;
    std::unique_ptr<dense::PackReader> reader_;   // the weights' blocks and the embedding (freed after the model)
    dense::ModelPtr model_;
};

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }

    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<Loaded>(); }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "qwen"}; }
    // Qwen2 / Qwen3's split (no combining marks in a letter run), as the checkpoint's tokenizer.json gives it
    PreTokenizer pretokenizer() const override { return {"qwen2", program::tokenizer::pretok_qwen2}; }
    model::TokenNames token_names() const override { return program::qwen_token_names(); }
    program::grammar::ToolForm tool_form() const override { return program::qwen_tool_form(); }
    program::GpuModes gpu_modes() const override { return {0, 0, 0}; }   // one card

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        (void) code;
        if (model::parse_rope_flag(a, args, o.rope)) {}
        else if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
        else if (a == "--kv") o.kv = args.next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(args.next("--kv-resident"));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(args.next("--suffix-draft")));
        else if (a == "--draft-log") basalt::core::diagnostics().draft_log = args.next("--draft-log");
        else return false;
        return true;
    }

    // Unless --spec says otherwise (0 or 1: plain decode): the pack's DFlash 2 block drafter drafts windows of up to
    // kBlockSpec (its trained block), else its MTP layer windows of up to kMtpSpec (each MTP draft reads the whole head, and
    // past the second position few are kept).  The measurements behind both: docs/models/sensenova-loop.md#drafting.
    int planned_spec(const program::CommonOptions& o) const override {
        if (o.spec_given) return o.spec >= 2 ? o.spec : 0;
        return default_spec(o.model);
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "fp16" && o.kv != "int8" && o.kv != "nvfp4") {
            basalt::log::error(kArea, "--kv %s: fp16 (the default), int8 or nvfp4", o.kv.c_str());
            return 2;
        }
        if (o.kv == "nvfp4" && o.kv_resident > 0) {
            basalt::log::error(kArea, "--kv nvfp4 keeps every cell on the card: leave --kv-resident off");
            return 2;
        }
        if (const auto c = model::check_rope_flags(o.rope, kArea)) return c;
        if (o.kv_resident < 0) {
            basalt::log::error(kArea, "--kv-resident %lld: cells per attention visit kept in VRAM, or 0 (all)", (long long) o.kv_resident);
            return 2;
        }
        if (o.spec_given && (o.spec < 0 || o.spec > 16)) {
            basalt::log::error(kArea, "--spec %d: 0 (off) to 16", o.spec);
            return 2;
        }
        if (!o.spec_given) o.spec = default_spec(o.model);
        if (o.spec >= 2 && default_spec(o.model) == 0) {
            basalt::log::warn(kArea, "--spec %d off: the pack has no drafter layer", o.spec);
            o.spec = 0;
        }
        return std::nullopt;
    }

private:
    static constexpr int kMtpSpec = 3, kBlockSpec = 8;
    // the window the pack's drafter fills by default: kBlockSpec with a block drafter, kMtpSpec with the MTP layer, else 0
    static int default_spec(const std::string& path) {
        core::ModelFile file;
        std::string err;
        if (path.empty() || !file.open(path, err) || !file.load_meta(err)) return 0;
        const core::Json* t = file.meta().get("tensors");
        if (t == nullptr) return 0;
        return t->get("dflash.fc.0.weight") ? kBlockSpec : t->get("mtp.fc.weight") ? kMtpSpec : 0;
    }
    const char* model_name() const override { return "SenseNova-Loop-9B"; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::sensenova_loop
