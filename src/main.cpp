#include <iostream>
#include <vector>

#include "args.h"
#include "nucleus/kernel.h"
#include "nucleus/model.h"
#include "reporter.h"
using namespace nucleus;
using namespace cli;

double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    Args a;
    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            NUC_CHECK(i + 1 < argc, "missing value for " + k);
            return argv[++i];
        };
        if (k == "--prompt")
            a.prompt = next();
        else if (k == "--chat")
            a.chat = true;
        else if (k == "--system")
            a.system = next();
        else if (k == "--max-tokens")
            a.max_tokens = std::stoi(next());
        else if (k == "--temp")
            a.temp = std::stof(next());
        else if (k == "--top-k")
            a.top_k = std::stoi(next());
        else if (k == "--top-p")
            a.top_p = std::stof(next());
        else if (k == "--seed")
            a.seed = std::stoull(next());
        else if (k == "--ctx")
            a.ctx = (u32) std::stoul(next());
        else if (k == "--max-rss")
            a.max_rss_gb = std::stod(next());
        else if (k == "--threads")
            a.threads = std::stoi(next());
        else if (k == "--show-specials")
            a.show_specials = true;
        else if (k == "-h" || k == "--help") {
            usage();
            return 0;
        } else
            pos_args.push_back(k);
    }
    if (pos_args.size() != 1) {
        usage();
        return 1;
    }
    a.model = pos_args[0];

    try {
        double t0 = now_s();
        Model m(a.model, a.ctx, a.threads);
        NUC_CHECK(a.max_rss_gb >= 0, "--max-rss must not be negative");
        const u64 max_rss = (u64) (a.max_rss_gb * (double) (1ull << 30));
        m.set_max_rss(max_rss);

        const MemBase mem_base{rss_bytes(), page_faults().major};
        std::fprintf(stderr,
                     "[nucleus] loaded %s in %.2fs | hidden=%u layers=%u "
                     "experts/layer=%u (top-%u) vocab=%u ctx=%u threads=%d%s | rss %s\n",
                     a.model.c_str(), now_s() - t0, m.cfg().hidden, m.cfg().n_layers, m.cfg().n_experts, m.cfg().top_k,
                     m.cfg().vocab, m.max_ctx(), a.threads, exact_mode() ? " [EXACT]" : "",
                     fmt_bytes(mem_base.rss).c_str());

        Sampler sampler(a.temp, a.top_k, a.top_p, a.seed);
        const Tokenizer &tok = m.tokenizer();
        u32 pos = 0;

        if (!a.chat) {
            NUC_CHECK(!a.prompt.empty(), "need --prompt or --chat");
            std::vector<i32> toks = tok.encode(a.prompt, /*add_bos=*/true, m.cfg().bos_id,
                                               /*parse_specials=*/true);
            double t1 = now_s();
            prefill(m, toks, pos);
            double t2 = now_s();
            i32 made = decode_loop(m, sampler, pos, a.max_tokens, a.show_specials);
            double t3 = now_s();
            std::printf("\n");
            std::fprintf(stderr,
                         "[nucleus] prefill %zu tok in %.2fs (%.2f tok/s) | "
                         "decode %d tok in %.2fs (%.2f tok/s)\n",
                         toks.size(), t2 - t1, toks.size() / (t2 - t1), made, t3 - t2, made / (t3 - t2));
            report_mem(mem_base, m, max_rss);
            return 0;
        }

        std::fprintf(stderr, "[nucleus] chat mode; empty line or Ctrl-D to quit\n");
        bool first = true;
        std::string line;
        for (;;) {
            std::fprintf(stderr, "\n> ");
            if (!std::getline(std::cin, line) || line.empty())
                break;

            std::string turn;
            if (first && !a.system.empty()) {
                turn += "<|turn>system\n" + a.system + "<turn|>\n";
            }
            if (!first)
                turn += "\n";
            turn += "<|turn>user\n" + line + "<turn|>\n<|turn>model\n";
            // TODO: find out how to think
            turn += "<|channel>thought\n<channel|>";

            std::vector<i32> toks = tok.encode(turn, /*add_bos=*/first, m.cfg().bos_id, true);
            double t1 = now_s();
            prefill(m, toks, pos);
            i32 made = decode_loop(m, sampler, pos, a.max_tokens, a.show_specials);
            double t2 = now_s();
            const u64 rss = rss_bytes();
            std::fprintf(stderr, "\n[%.2f tok/s | rss %s, %s since load]\n", (toks.size() + made) / (t2 - t1),
                         fmt_bytes(rss).c_str(), fmt_delta((i64) rss - (i64) mem_base.rss).c_str());
            first = false;
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
