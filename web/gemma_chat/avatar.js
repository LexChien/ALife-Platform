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

  function setState(state) {
    const wrap = document.getElementById("avatarWrap");
    if (wrap) wrap.dataset.state = state;
  }

  preload();
  window.DigiAvatar = { attach, setState, setLevel, isReady: () => ready, face };
})();
