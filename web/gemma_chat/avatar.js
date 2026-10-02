// Plan 38 J3: lip-sync on the FIXED avatar. Mouth keyframes k0..k7 are pre-rendered offline from avatar.jpg
// (mouth ROI only, identity-guard checked: runs/plan38/avatar/mouth/guard_report.json). Nothing restyles the face:
// the mouth layer is an identical-geometry copy whose only difference is the mouth region.
(function () {
  const ATTACK = 0.6, RELEASE = 0.25, LEVELS = 8;
  const face = document.getElementById("avatarFace");
  const mouth = document.getElementById("avatarMouth");
  const frames = [];
  let ready = false, ctx = null, analyser = null, buf = null, level = 0, raf = null, active = 0;
  let gain = 1 / 0.12; // envelope normaliser; adapts to the loudest recent frame

  function preload() {
    let loaded = 0;
    for (let i = 0; i < LEVELS; i++) {
      const im = new Image();
      im.onload = () => { loaded += 1; if (loaded === LEVELS) ready = true; };
      im.onerror = () => { ready = false; };
      im.src = `/avatar/mouth/k${i}.jpg`;
      frames.push(im.src);
    }
  }

  function ensureCtx() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      buf = new Float32Array(analyser.fftSize);
      analyser.connect(ctx.destination);
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  function tick() {
    raf = null;
    if (!analyser || active <= 0) { setLevel(0); return; }
    analyser.getFloatTimeDomainData(buf);
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    const rms = Math.sqrt(s / buf.length);
    gain = Math.min(gain, 1 / Math.max(rms, 0.02)) * 1.002;
    const target = Math.min(1, rms * gain);
    level += (target > level ? ATTACK : RELEASE) * (target - level);
    setLevel(level);
    raf = requestAnimationFrame(tick);
  }

  function setLevel(v) {
    if (!mouth) return;
    const k = Math.max(0, Math.min(LEVELS - 1, Math.round(v * (LEVELS - 1))));
    if (!ready || k === 0) { mouth.style.opacity = "0"; return; }
    const src = frames[k];
    if (mouth.dataset.k !== String(k)) { mouth.src = src; mouth.dataset.k = String(k); }
    mouth.style.opacity = "1";
  }

  // Route an <audio> element through the analyser (audio still plays normally).
  function attach(audio) {
    try {
      ensureCtx();
      if (!audio._digiSrc) {
        audio.crossOrigin = "anonymous";
        audio._digiSrc = ctx.createMediaElementSource(audio);
        audio._digiSrc.connect(analyser);
      }
      audio.addEventListener("play", () => { active += 1; if (!raf) raf = requestAnimationFrame(tick); });
      const done = () => { active = Math.max(0, active - 1); };
      audio.addEventListener("ended", done);
      audio.addEventListener("pause", done);
    } catch (e) {
      console.warn("lip-sync attach failed", e);
    }
  }

  // ---- J3.3 expression clips (only clips whose identity guard PASSed are in the manifest as installed)
  const clipEl = document.getElementById("avatarClip");
  let clips = {}, curState = "idle", curEmotion = "neutral", curClip = null;
  fetch("/avatar/clips/manifest.json").then((r) => (r.ok ? r.json() : { clips: {} })).then((m) => {
    for (const [name, c] of Object.entries(m.clips || {})) if (c && c.guard_pass && c.installed) clips[name] = `/avatar/clips/${name}.mp4`;
    applyClip();
  }).catch(() => {});

  // Plan 40: profile 'yaying' adds her own-motion clips (idle sway / smile / glance / head tilt / speaking loop);
  // profile 'digiclone' keeps the exact J3 mapping (smile / concerned / idle, no clip while speaking).
  let profile = "digiclone";
  fetch("/api/profile").then((r) => (r.ok ? r.json() : null)).then((p) => { if (p && p.active) { profile = p.active; applyClip(); } }).catch(() => {});

  function pickClip(state, emotion, prof, have) {
    const yy = prof === "yaying";
    if (state === "speaking") return yy && have.yaying_speaking ? "yaying_speaking" : null;
    if (emotion === "smile") return yy && have.yaying_smile ? "yaying_smile" : (have.smile ? "smile" : (have.idle ? "idle" : null));
    if (emotion === "concerned" && have.concerned) return "concerned";
    if (yy && (emotion === "attentive" || state === "thinking") && have.yaying_glance) return "yaying_glance";
    if (yy && (emotion === "reassuring" || emotion === "calm" || state === "listening") && have.yaying_head_tilt) return "yaying_head_tilt";
    if (yy && have.yaying_idle_sway) return "yaying_idle_sway";
    return have.idle ? "idle" : null;
  }

  function applyClip() {
    if (!clipEl) return;
    const want = pickClip(curState, curEmotion, profile, clips);
    const wrapEl = document.getElementById("avatarWrap");
    if (wrapEl) wrapEl.dataset.clip = want || "";
    if (want === curClip) return;
    curClip = want;
    if (!want) { clipEl.classList.remove("on"); return; }
    clipEl.classList.remove("on");
    setTimeout(() => {
      clipEl.src = clips[want];
      const p = clipEl.play();
      const show = () => clipEl.classList.add("on");
      if (p && p.then) p.then(show).catch(() => {}); else show();
    }, 150);
  }

  function setState(state) {
    const wrap = document.getElementById("avatarWrap");
    if (wrap) wrap.dataset.state = state;
    curState = state || "idle";
    applyClip();
  }

  function setEmotion(e) { curEmotion = e || "neutral"; applyClip(); }

  preload();
  window.DigiAvatar = { attach, setState, setEmotion, setLevel, pickClip,
    setProfile: (p) => { profile = p || "digiclone"; applyClip(); }, profile: () => profile, isReady: () => ready, face, clips: () => ({ ...clips }), current: () => curClip };
})();
