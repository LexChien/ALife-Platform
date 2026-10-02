"""Plan 38 J1.1/J1.2: llama-server adapter against a mock SSE server, manager flags, and a real smoke (skipped
when no llama-server is listening on :8091)."""
import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from genai.llm.adapter import LLMRequest
from genai.llm.backends.llama_server import LlamaServerAdapter, LlamaServerError
from genai.llm.server_manager import LlamaServerManager, ServerConfig


class _Mock(BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        _Mock.bodies.append(req)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("你好", "，Lex", "。"):
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "timings": {"prompt_n": 7, "cache_n": 30, "predicted_n": 3, "predicted_per_second": 90.0}}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())


class _Fallback:
    backend_name = "llama_cpp"

    def generate(self, request):
        from genai.llm.adapter import LLMResponse
        return LLMResponse(text="fallback", model_family="gemma", backend="llama_cpp", runtime={})


class LlamaServerAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_stream_and_timings(self):
        a = LlamaServerAdapter(url=self.url)
        deltas = []
        out = a.chat([{"role": "user", "content": "hi"}], on_delta=deltas.append, slot=1,
                     json_schema={"type": "object"}, max_tokens=12)
        self.assertEqual(out["text"], "你好，Lex。")
        self.assertEqual(deltas, ["你好", "，Lex", "。"])
        self.assertEqual(out["prompt_n"], 7)
        self.assertEqual(out["cache_n"], 30)
        self.assertIsNotNone(out["ttft_s"])
        body = _Mock.bodies[-1]
        self.assertEqual(body["id_slot"], 1)
        self.assertTrue(body["cache_prompt"])
        self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_generate_puts_dynamic_context_last(self):
        a = LlamaServerAdapter(url=self.url)
        req = LLMRequest(prompt="現在幾點？", system="STABLE PERSONA", context="memory: X",
                         metadata={"history": [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "你好"}]})
        r = a.generate(req)
        self.assertEqual(r.backend, "llama_server")
        msgs = _Mock.bodies[-1]["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "STABLE PERSONA"})
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertTrue(msgs[-1]["content"].startswith("memory: X"))
        self.assertTrue(msgs[-1]["content"].endswith("現在幾點？"))

    def test_unreachable_raises_or_falls_back(self):
        a = LlamaServerAdapter(url="http://127.0.0.1:9", timeout=1)
        with self.assertRaises(LlamaServerError):
            a.chat([{"role": "user", "content": "x"}])
        b = LlamaServerAdapter(url="http://127.0.0.1:9", timeout=1, fallback=_Fallback())
        self.assertEqual(b.generate(LLMRequest(prompt="x")).text, "fallback")
        self.assertFalse(b.healthcheck()["ok"])

    def test_manager_flags(self):
        cmd = ServerConfig().command()
        self.assertIn("-rea", cmd)
        self.assertEqual(cmd[cmd.index("-rea") + 1], "off")
        self.assertNotIn("--reasoning-budget", cmd)  # Plan 38: budget 0 alone leaks; -rea off is the switch
        for flag in ("--jinja", "--metrics", "-np", "--cache-reuse"):
            self.assertIn(flag, cmd)
        self.assertIn("--reasoning-budget", ServerConfig(reasoning_budget_zero=True).command())

    def test_factory_selects_llama_server(self):
        from genai.llm.factory import create_llm_adapter
        a = create_llm_adapter({"llm": {"backend": "llama_server", "model_family": "gemma",
                                        "server": {"manage": False, "url": self.url, "fallback": "none"}}})
        self.assertEqual(a.backend_name, "llama_server")
        self.assertTrue(a.healthcheck()["ok"])


def _real_up():
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=1).read()).get("status") == "ok"
    except Exception:
        return False


@unittest.skipUnless(_real_up(), "no real llama-server on :8091")
class RealServerSmokeTest(unittest.TestCase):
    def test_real_short_answer_no_leak(self):
        from cognition.leak_guard import check_leak
        a = LlamaServerAdapter(url="http://127.0.0.1:8091")
        out = a.chat([{"role": "system", "content": "你是冷靜的 AI 管家，以繁體中文簡短回答。"},
                      {"role": "user", "content": "17 乘以 23 等於多少？"}], max_tokens=40, temperature=0.0)
        self.assertIn("391", out["text"])
        self.assertFalse(check_leak(out["text"]).leak)
        self.assertLess(out["ttft_s"], 5.0)
        self.assertTrue(LlamaServerManager(ServerConfig()).is_healthy())


if __name__ == "__main__":
    unittest.main()
