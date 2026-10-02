// Plan 38 J1.3/J2: realtime voice client. Mic -> AudioWorklet (16 kHz Int16, 20 ms) -> ws://host:port+1/ws/session.
// Modes: open mic / wake word (Hey Jarvis, 你好管家) / push-to-talk. PTT (button or hold Space) works in every mode.
// Server events drive a sequential audio queue (ack first, then sentences), barge-in stop, HUD and lip-sync.
(function () {
  const $ = (id) => document.getElementById(id);
  const statusEl = $("rtStatus"), modeSel = $("rtMode"), connectBtn = $("rtConnect"), pttBtn = $("rtPtt");
  const transcriptEl = $("rtTranscript");
  let ws = null, ctx = null, node = null, stream = null, sessionId = null;
  let queue = [], playing = null, turnId = null, stopT = null, firstPlayedReported = {};

  function wsUrl() {
    const port = (parseInt(location.port || "80", 10) + 1);
    return `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${port}/ws/session`;
  }
  function setStatus(t) { if (statusEl) statusEl.textContent = t; }
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

  // ---------------------------------------------------------------- playback queue
  function enqueue(url, kind, tid) { queue.push({ url, kind, tid }); if (!playing) playNext(); }
  function playNext() {
    const item = queue.shift();
    if (!item) { playing = null; send({ type: "playback", event: "ended", turn_id: turnId }); return; }
    const a = new Audio(item.url);
    playing = a;
    if (window.DigiAvatar) window.DigiAvatar.attach(a);
    a.onplay = () => {
      if (item.kind === "sentence" && !firstPlayedReported[item.tid]) {
        firstPlayedReported[item.tid] = true;
        send({ type: "playback", event: "started", turn_id: item.tid });
      }
    };
    a.onended = playNext;
    a.onerror = playNext;
    const p = a.play();
    if (p && p.catch) p.catch(playNext);
  }
  function stopAll(why, detectMs) {
    const t0 = performance.now();
    queue = [];
    if (playing) { playing.pause(); playing = null; }
    send({ type: "playback", event: "stopped", turn_id: turnId, latency_ms: Math.round(performance.now() - t0), why, detect_ms: detectMs });
  }

  // ---------------------------------------------------------------- events
  function onEvent(ev) {
    const hud = window.DigiHUD;
    switch (ev.type) {
      case "state":
        if (hud) hud.setState(ev.state, ev.why || "");
        if (window.DigiAvatar) window.DigiAvatar.setState(ev.state);
        break;
      case "wake": setStatus(`WAKE: ${ev.keyword}`); break;
      case "vad": if (hud && ev.event === "speech_start") hud.setState("listening"); break;
      case "transcript":
        if (transcriptEl) transcriptEl.textContent = ev.text;
        if (window.appendMessage && ev.text) window.appendMessage("user", ev.text);
        break;
      case "ack": enqueue(ev.url, "ack", ev.turn_id); break;
      case "audio": turnId = ev.turn_id; enqueue(ev.url, "sentence", ev.turn_id); break;
      case "thought": if (hud) hud.addThought(ev.thought || ev); break;
      case "emotion":
        if (hud && ev.emotion && ev.emotion.state) hud.showMood(ev.emotion.state);
        if (window.DigiAvatar && ev.emotion && ev.emotion.modulation && ev.emotion.modulation.avatar)
          window.DigiAvatar.setEmotion(ev.emotion.modulation.avatar.expression);
        break;
      case "stop": stopAll(ev.why, ev.detect_ms); break;
      case "done":
        sessionId = ev.session_id || sessionId;
        if (window.appendMessage && ev.reply && !ev.cancelled) window.appendMessage("assistant", ev.reply);
        if (hud && ev.self_state) hud.showMood(ev.self_state);
        break;
      case "trace": if (hud) hud.showTrace(ev); break;
      case "error": setStatus(`ERROR ${ev.error}`); break;
    }
  }

  // ---------------------------------------------------------------- connect
  async function connect() {
    if (ws) { disconnect(); return; }
    ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    ws.onopen = async () => {
      setStatus("CONNECTED");
      connectBtn.textContent = "DISCONNECT";
      send({ type: "hello", session_id: sessionId, mode: modeSel.value, thoughts: window.DigiHUD ? window.DigiHUD.thoughtsOn() : false });
      try { await startMic(); } catch (e) { setStatus(`MIC ERROR ${e.message}`); }
    };
    ws.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => { setStatus("DISCONNECTED"); connectBtn.textContent = "LIVE VOICE"; stopMic(); ws = null; };
    ws.onerror = () => setStatus("WS ERROR (is the realtime port up?)");
  }
  function disconnect() { if (ws) ws.close(); }

  async function startMic() {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "pcm-uploader");
    node.port.onmessage = (e) => { if (ws && ws.readyState === 1) ws.send(e.data); };
    src.connect(node);
  }
  function stopMic() {
    if (node) node.disconnect();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ctx) ctx.close();
    node = stream = ctx = null;
  }

  // ---------------------------------------------------------------- controls
  if (connectBtn) connectBtn.addEventListener("click", connect);
  if (modeSel) modeSel.addEventListener("change", () => send({ type: "set", mode: modeSel.value }));
  const pttDown = () => { if (playing) stopAll("ptt"); send({ type: "ptt", down: true }); pttBtn && pttBtn.classList.add("active"); };
  const pttUp = () => { send({ type: "ptt", down: false }); pttBtn && pttBtn.classList.remove("active"); };
  if (pttBtn) {
    pttBtn.addEventListener("pointerdown", pttDown);
    pttBtn.addEventListener("pointerup", pttUp);
    pttBtn.addEventListener("pointerleave", () => pttBtn.classList.contains("active") && pttUp());
  }
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && ws && document.activeElement && document.activeElement.tagName !== "TEXTAREA") { e.preventDefault(); pttDown(); }
    if (e.code === "Escape" && ws) { stopAll("esc"); send({ type: "barge_in" }); }
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space" && ws && document.activeElement && document.activeElement.tagName !== "TEXTAREA") { e.preventDefault(); pttUp(); }
  });
  if (window.DigiHUD) window.DigiHUD.onThoughtToggle((on) => send({ type: "set", thoughts: on }));
  window.DigiRealtime = { connect, send, isConnected: () => !!(ws && ws.readyState === 1), sendText: (t) => send({ type: "text", text: t }) };
})();
