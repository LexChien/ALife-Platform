// Plan 38 J4.4: JARVIS-style HUD (state, latency trace, mood, private thought stream). Thought stream is OFF by default.
(function () {
  const $ = (id) => document.getElementById(id);
  const thoughtToggle = $("thoughtToggle");
  const thoughtList = $("hudThoughts");
  const traceEl = $("hudTrace");
  const stateEl = $("hudState");
  const moodEl = $("hudMood");
  const listeners = [];
  if (thoughtToggle) thoughtToggle.checked = false; // default OFF, never persisted ON implicitly

  function setState(state, extra) {
    if (stateEl) stateEl.textContent = (state || "idle").toUpperCase() + (extra ? ` · ${extra}` : "");
    document.body.dataset.hudState = state || "idle";
  }

  function fmt(v) { return v == null ? "--" : `${Math.round(v * 1000)}ms`; }

  function showTrace(t) {
    if (!traceEl || !t || !t.marks) return;
    const m = t.marks;
    const rows = [["STT", m.stt_done], ["ACK", m.ack_audio], ["1st sent", m.first_sentence],
                  ["1st audio", m.first_audio_ready], ["played", m.first_audio_played], ["done", m.turn_done]];
    traceEl.innerHTML = rows.map(([k, v]) => `<span><b>${k}</b> ${fmt(v)}</span>`).join("");
  }

  function showMood(state) {
    if (!moodEl || !state) return;
    const keys = ["valence", "arousal", "energy", "curiosity", "concern"];
    moodEl.innerHTML = keys.filter((k) => typeof state[k] === "number").map((k) => {
      const v = state[k];
      const pct = Math.round(((k === "valence" ? (v + 1) / 2 : v)) * 100);
      return `<div class="mood-row"><span>${k}</span><i style="width:${Math.max(2, Math.min(100, pct))}%"></i></div>`;
    }).join("") + (state.feeling ? `<div class="mood-feeling">${state.feeling}</div>` : "");
  }

  function addThought(th) {
    if (!thoughtList || !thoughtToggle || !thoughtToggle.checked || !th) return;
    const li = document.createElement("li");
    const text = th.private_note || th.summary || th.note || "";
    li.textContent = `${th.feeling ? "[" + th.feeling + "] " : ""}${text}`;
    thoughtList.prepend(li);
    while (thoughtList.children.length > 12) thoughtList.removeChild(thoughtList.lastChild);
  }

  if (thoughtToggle) {
    thoughtToggle.addEventListener("change", () => {
      if (thoughtList) {
        thoughtList.hidden = !thoughtToggle.checked;
        if (!thoughtToggle.checked) thoughtList.innerHTML = "";
      }
      listeners.forEach((fn) => fn(thoughtToggle.checked));
    });
  }
  window.DigiHUD = { setState, showTrace, showMood, addThought, onThoughtToggle: (fn) => listeners.push(fn),
                     thoughtsOn: () => !!(thoughtToggle && thoughtToggle.checked) };
})();
