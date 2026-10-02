const chatLog = document.getElementById("chatLog");
const messageInput = document.getElementById("messageInput");
const composerForm = document.getElementById("composerForm");
const sendButton = document.getElementById("sendButton");
const micButton = document.getElementById("micButton");
const resetButton = document.getElementById("resetButton");
const speakToggle = document.getElementById("speakToggle");
const autoSubmitToggle = document.getElementById("autoSubmitToggle") || document.getElementById("submitToggle");
const serviceStatus = document.getElementById("serviceStatus");
const backendStatus = document.getElementById("backendStatus");
const sessionStatus = document.getElementById("sessionStatus");
const avatarStatus = document.getElementById("avatarStatus");
const micHint = document.getElementById("micHint");
const avatarWrap = document.getElementById("avatarWrap");
const avatarStage = document.getElementById("avatarStage");
const messageTemplate = document.getElementById("messageTemplate");
const userVideo = document.getElementById("userVideo");
const userVideoWrap = document.getElementById("userVideoWrap");
const cameraToggle = document.getElementById("cameraToggle");
const lifeVisual = document.getElementById("lifeVisual");
const lifeFallback = document.getElementById("lifeFallback");
const lifeSummary = document.getElementById("lifeSummary");
const lifeRunStatus = document.getElementById("lifeRunStatus");
const lifeScore = document.getElementById("lifeScore");
const lifePhase = document.getElementById("lifePhase");
const lifeMilestones = document.getElementById("lifeMilestones");
const lifePhases = document.getElementById("lifePhases");
const lifeRuns = document.getElementById("lifeRuns");
const lifeTasks = document.getElementById("lifeTasks");
const emotionStatus = document.getElementById("emotionStatus");
const dnaStatus = document.getElementById("dnaStatus");
let serverTtsAvailable = false;
let currentAudio = null;
let lastVoiceModulation = null;
const quickButtons = Array.from(document.querySelectorAll(".quick-button"));

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const storageKey = "gemma-web-session-id";
const audioCaptureConstraints = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
};

let sessionId = localStorage.getItem(storageKey) || crypto.randomUUID();
let recognition = null;
let recognitionActive = false;
let mediaRecorder = null;
let mediaRecorderChunks = [];
let mediaRecorderStream = null;
let mediaRecorderActive = false;
let mediaRecorderStartedAt = 0;
let mediaRecorderPeakLevel = 0;
let audioMonitorContext = null;
let audioMonitorSource = null;
let audioMonitorTimer = null;
let audioMonitorData = null;
let microphonePermission = "unknown";
let cameraStream = null;
let cameraActive = false;
let voiceConfig = {
  recognition_lang: "zh-TW",
  synthesis_lang: "zh-TW",
  auto_submit: true,
  auto_speak: true,
};
let avatarConfig = {
  states: {
    idle: "idle",
    listening: "listening",
    thinking: "thinking",
    speaking: "speaking",
    error: "error",
  },
};
let conversationState = "ready";
let micState = "mic_ready";
let speechState = "speech_ready";
let hasErrorState = false;

localStorage.setItem(storageKey, sessionId);
if (sessionStatus) {
  sessionStatus.textContent = sessionId;
}

function setAvatarState(state) {
  if (avatarWrap) {
    avatarWrap.dataset.state = avatarConfig.states?.[state] || state;
  }
  if (avatarStatus) {
    avatarStatus.textContent = state;
  }
}

function syncAvatarState() {
  let state = "idle";
  if (hasErrorState) {
    state = "error";
  } else if (micState === "mic_listening") {
    state = "listening";
  } else if (conversationState === "submitting" || conversationState === "awaiting_response") {
    state = "thinking";
  } else if (speechState === "speech_playing") {
    state = "speaking";
  }
  setAvatarState(state);
}

function setConversationState(nextState) {
  conversationState = nextState;
  hasErrorState = nextState === "failed";
  syncAvatarState();
}

function setMicState(nextState) {
  micState = nextState;
  hasErrorState = nextState === "mic_error" ? true : hasErrorState && conversationState === "failed";
  syncAvatarState();
}

function setSpeechState(nextState) {
  speechState = nextState;
  hasErrorState = nextState === "speech_error" ? true : hasErrorState && conversationState === "failed";
  syncAvatarState();
}

function clearErrorState() {
  hasErrorState = false;
  if (conversationState === "failed") {
    conversationState = "ready";
  }
  if (micState === "mic_error") {
    micState = recognition ? "mic_ready" : "mic_unavailable";
  }
  if (speechState === "speech_error") {
    speechState = "speech_ready";
  }
  syncAvatarState();
}

function appendMessage(role, text) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role === "user" ? "user-row" : "assistant-row");
  const bubble = node.querySelector(".message-bubble");
  bubble.textContent = text;
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendSystemMessage(text) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add("system-row");
  const bubble = node.querySelector(".message-bubble");
  bubble.textContent = text;
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setComposerBusy(isBusy) {
  sendButton.disabled = isBusy;
  micButton.disabled = isBusy && !recognitionActive && !mediaRecorderActive;
  messageInput.disabled = isBusy;
}

function supportsServerRecording() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
}

function isServerRecordingMode() {
  return voiceConfig.input_provider === "macos_speech_via_upload" || voiceConfig.input_provider === "local_whisper_via_upload";
}

function preferredRecordingMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];
  for (const mimeType of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(mimeType)) {
      return mimeType;
    }
  }
  return "";
}

function formatMicError(error) {
  if (!error) {
    return "unknown_error";
  }
  if (typeof error === "string") {
    return error;
  }
  return error.message || error.name || String(error);
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(3);
}

function createTextNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  node.textContent = text;
  return node;
}

function renderLife(payload) {
  if (!payload || !payload.enabled) {
    if (lifeRunStatus) {
      lifeRunStatus.textContent = "DISABLED";
    }
    if (lifeSummary) {
      lifeSummary.textContent = payload?.summary || "ASAL 進度整合未啟用";
    }
    return;
  }

  const latest = payload.latest_run || {};
  const details = latest.details || {};
  const metrics = latest.metrics || {};
  const assetUrls = latest.asset_urls || {};
  const visualUrl = assetUrls.gif || assetUrls.image || assetUrls.keyframe_fusion || assetUrls.keyframe_birth;

  if (lifeRunStatus) {
    lifeRunStatus.textContent = latest.run_id ? `${latest.run_id} / ${latest.status || "unknown"}` : "NO RUN";
  }
  if (lifeSummary) {
    lifeSummary.textContent = payload.summary || "尚未找到 ASAL 進度摘要";
  }
  if (lifeScore) {
    lifeScore.textContent = `score ${formatScore(metrics.best_score)} / narrative ${formatScore(details.narrative_score)}`;
  }
  if (lifePhase) {
    const actual = Array.isArray(details.actual_component_sequence)
      ? details.actual_component_sequence.join("→")
      : "--";
    const valid = details.narrative_phase_order_valid === true ? "valid" : "unverified";
    lifePhase.textContent = `phase ${actual} / ${valid}`;
  }
  if (lifeVisual && visualUrl) {
    lifeVisual.src = visualUrl;
    lifeVisual.hidden = false;
    if (lifeFallback) {
      lifeFallback.hidden = true;
    }
  }
  if (lifeMilestones) {
    lifeMilestones.innerHTML = "";
    const capabilities = Array.isArray(payload.capabilities) ? payload.capabilities : [];
    for (const item of capabilities.slice(0, 6)) {
      const node = document.createElement("span");
      node.className = `life-chip ${item.state === "done" ? "done" : "pending"}`;
      node.textContent = item.label;
      lifeMilestones.appendChild(node);
    }
  }
  if (lifePhases) {
    lifePhases.innerHTML = "";
    const phases = Array.isArray(details.phase_scores) ? details.phase_scores : [];
    for (const phase of phases) {
      const card = document.createElement("article");
      card.className = "phase-card";
      if (phase.keyframe_url) {
        const image = document.createElement("img");
        image.src = phase.keyframe_url;
        image.alt = `${phase.name} phase keyframe`;
        card.appendChild(image);
      }
      card.appendChild(createTextNode("span", "phase-name", String(phase.name || "phase")));
      card.appendChild(
        createTextNode(
          "span",
          "phase-score",
          `score ${formatScore(phase.score)} / target ${phase.target_components ?? "--"}`,
        ),
      );
      lifePhases.appendChild(card);
    }
  }
  if (lifeRuns) {
    lifeRuns.innerHTML = "";
    const recentRuns = Array.isArray(payload.recent_runs) ? payload.recent_runs : [];
    for (const run of recentRuns.slice(0, 4)) {
      const row = document.createElement("div");
      row.className = "run-row";
      row.appendChild(createTextNode("span", "run-id", run.run_id || "--"));
      row.appendChild(createTextNode("span", "run-score", formatScore(run.metrics?.best_score)));
      row.appendChild(createTextNode("span", "run-time", run.timeline_label || ""));
      lifeRuns.appendChild(row);
    }
  }
  if (lifeTasks) {
    lifeTasks.innerHTML = "";
    const pending = Array.isArray(payload.tasks?.pending) ? payload.tasks.pending : [];
    for (const item of pending.slice(0, 3)) {
      lifeTasks.appendChild(createTextNode("div", "task-item", item));
    }
    if (!pending.length) {
      lifeTasks.appendChild(createTextNode("div", "task-item", "no open item in TASKS.md"));
    }
  }
}

async function queryMicrophonePermission() {
  if (!navigator.permissions || !navigator.permissions.query) {
    return "unknown";
  }
  try {
    const result = await navigator.permissions.query({ name: "microphone" });
    if (!result || !result.state) {
      return "unknown";
    }
    microphonePermission = result.state;
    result.onchange = () => {
      microphonePermission = result.state;
      if (result.state === "granted") {
        micHint.textContent = "麥克風權限已允許，可開始語音輸入";
        if (micState !== "mic_listening") {
          setMicState("mic_ready");
        }
      }
    };
    return result.state;
  } catch (_) {
    return "unknown";
  }
}

async function ensureMicrophoneAccess(promptUser = false, keepStream = false) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return { ok: false, reason: "media_devices_unavailable" };
  }

  if (!promptUser) {
    const state = await queryMicrophonePermission();
    return { ok: state === "granted", reason: state };
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia(audioCaptureConstraints);
    if (keepStream) {
      microphonePermission = "granted";
      return { ok: true, reason: "granted", stream };
    }
    stream.getTracks().forEach((track) => track.stop());
    microphonePermission = "granted";
    return { ok: true, reason: "granted" };
  } catch (error) {
    const reason = formatMicError(error);
    microphonePermission = reason === "NotAllowedError" ? "denied" : "unknown";
    return { ok: false, reason };
  }
}

function renderEmotion(emotion, dna) {
  if (emotion && emotion.state) {
    const st = emotion.state;
    if (emotionStatus) {
      emotionStatus.textContent = `${st.label} v${Number(st.valence).toFixed(2)} a${Number(st.arousal).toFixed(2)}`;
    }
    if (avatarStage && emotion.modulation && emotion.modulation.avatar) {
      // Plan 38 J0.3: colour the halo, never the face.
      avatarStage.dataset.emotion = emotion.modulation.avatar.expression || "neutral";
    }
    lastVoiceModulation = emotion.modulation ? emotion.modulation.tts : null;
  }
  if (dna && dnaStatus) {
    dnaStatus.textContent = `${dna.genome_id} g${dna.generation}`;
  }
}

function playServerAudio(url, fallbackText) {
  try {
    if (currentAudio) {
      currentAudio.pause();
    }
    currentAudio = new Audio(url);
    currentAudio.onplay = () => {
      setConversationState("reply_received");
      setSpeechState("speech_playing");
    };
    currentAudio.onended = () => {
      setSpeechState(speakToggle.checked ? "speech_ready" : "speech_disabled");
      setConversationState("ready");
    };
    currentAudio.onerror = () => {
      speakText(fallbackText);
    };
    const playing = currentAudio.play();
    if (playing && playing.catch) {
      playing.catch(() => speakText(fallbackText));
    }
  } catch (error) {
    speakText(fallbackText);
  }
}

// Plan 37 R2: play sentence chunks in order; chunk 0 is ready immediately, later chunks are
// synthesized server-side while earlier ones play (GET waits until a pending chunk exists).
function playServerAudioQueue(urls, fallbackText) {
  if (currentAudio) {
    currentAudio.pause();
  }
  let index = 0;
  const playNext = () => {
    if (index >= urls.length) {
      setSpeechState(speakToggle.checked ? "speech_ready" : "speech_disabled");
      setConversationState("ready");
      return;
    }
    const audio = new Audio(urls[index]);
    currentAudio = audio;
    if (index === 0) {
      audio.onplay = () => {
        setConversationState("reply_received");
        setSpeechState("speech_playing");
      };
    }
    audio.onended = () => {
      index += 1;
      playNext();
    };
    audio.onerror = () => {
      if (index === 0) {
        speakText(fallbackText);
      } else {
        index += 1;
        playNext();
      }
    };
    const playing = audio.play();
    if (playing && playing.catch) {
      playing.catch(() => (index === 0 ? speakText(fallbackText) : null));
    }
  };
  playNext();
}

function deliverReply(payload) {
  appendMessage("assistant", payload.reply);
  renderLife(payload.life);
  renderEmotion(payload.emotion, payload.dna);
  if (sessionStatus) {
    sessionStatus.textContent = payload.session_id;
  }
  if (speakToggle.checked) {
    if (payload.tts && payload.tts.ok && Array.isArray(payload.tts.chunks) && payload.tts.chunks.length > 1) {
      playServerAudioQueue(payload.tts.chunks.map((c) => c.url), payload.reply);
    } else if (payload.tts && payload.tts.ok && payload.tts.url) {
      playServerAudio(payload.tts.url, payload.reply);
    } else {
      speakText(payload.reply);
    }
  } else {
    setConversationState("reply_received");
    setConversationState("ready");
  }
}

async function uploadRecordingForVoiceChat(blob) {
  if (!blob || blob.size === 0) {
    throw new Error("empty_recording");
  }
  const response = await fetch("/api/voice_chat", {
    method: "POST",
    headers: {
      "Content-Type": blob.type || "application/octet-stream",
      "X-Session-ID": sessionId,
    },
    body: blob,
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function uploadRecordingForTranscription(blob) {
  if (!blob || blob.size === 0) {
    throw new Error("empty_recording");
  }
  const response = await fetch("/api/transcribe", {
    method: "POST",
    headers: {
      "Content-Type": blob.type || "application/octet-stream",
      "X-Session-ID": sessionId,
    },
    body: blob,
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function cleanupRecorderStream() {
  stopAudioLevelMonitor();
  if (mediaRecorderStream) {
    mediaRecorderStream.getTracks().forEach((track) => track.stop());
  }
  mediaRecorderStream = null;
  mediaRecorder = null;
  mediaRecorderChunks = [];
  mediaRecorderActive = false;
  mediaRecorderStartedAt = 0;
  mediaRecorderPeakLevel = 0;
}

function stopAudioLevelMonitor() {
  if (audioMonitorTimer) {
    window.clearInterval(audioMonitorTimer);
  }
  audioMonitorTimer = null;
  audioMonitorSource = null;
  audioMonitorData = null;
  if (audioMonitorContext) {
    audioMonitorContext.close().catch(() => {});
  }
  audioMonitorContext = null;
}

function startAudioLevelMonitor(stream) {
  stopAudioLevelMonitor();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    return;
  }
  audioMonitorContext = new AudioContextClass();
  audioMonitorContext.resume().catch(() => {});
  audioMonitorSource = audioMonitorContext.createMediaStreamSource(stream);
  const analyser = audioMonitorContext.createAnalyser();
  analyser.fftSize = 1024;
  audioMonitorData = new Uint8Array(analyser.fftSize);
  audioMonitorSource.connect(analyser);
  mediaRecorderPeakLevel = 0;
  audioMonitorTimer = window.setInterval(() => {
    analyser.getByteTimeDomainData(audioMonitorData);
    let sum = 0;
    for (const sample of audioMonitorData) {
      const centered = (sample - 128) / 128;
      sum += centered * centered;
    }
    const rms = Math.sqrt(sum / audioMonitorData.length);
    mediaRecorderPeakLevel = Math.max(mediaRecorderPeakLevel, rms);
    if (micState === "mic_listening") {
      const signal = rms >= 0.012 ? "收到聲音" : "收音偏弱";
      micHint.textContent = `正在錄音：${signal}，說完後再按一次停止`;
    }
  }, 180);
}

function hasEnoughRecordedSignal() {
  const elapsedMs = mediaRecorderStartedAt ? Date.now() - mediaRecorderStartedAt : 0;
  if (elapsedMs < 700) {
    return true;
  }
  if (!audioMonitorContext) {
    return true;
  }
  return mediaRecorderPeakLevel >= 0.008;
}

async function startServerRecording() {
  if (!supportsServerRecording()) {
    micHint.textContent = "此瀏覽器不支援錄音上傳模式，請改用最新 Chrome 或 Edge";
    setMicState("mic_error");
    return;
  }

  const micAccess = await ensureMicrophoneAccess(true, true);
  if (!micAccess.ok) {
    micHint.textContent = `麥克風權限或裝置不可用：${micAccess.reason}`;
    setMicState("mic_error");
    return;
  }

  const mimeType = preferredRecordingMimeType();
  mediaRecorderChunks = [];
  mediaRecorderStream = micAccess.stream;
  startAudioLevelMonitor(mediaRecorderStream);
  mediaRecorder = mimeType
    ? new MediaRecorder(mediaRecorderStream, { mimeType })
    : new MediaRecorder(mediaRecorderStream);

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      mediaRecorderChunks.push(event.data);
    }
  };

  mediaRecorder.onstart = () => {
    mediaRecorderActive = true;
    mediaRecorderStartedAt = Date.now();
    stopSpeechPlayback();
    clearErrorState();
    micButton.textContent = "⏹️ 停止錄音";
    micHint.textContent = "正在錄音，說完後再按一次停止";
    setMicState("mic_listening");
  };

  mediaRecorder.onerror = (event) => {
    micHint.textContent = `錄音失敗：${formatMicError(event.error || event.name || event)}`;
    cleanupRecorderStream();
    setMicState("mic_error");
  };

  mediaRecorder.onstop = async () => {
    micButton.textContent = "🎙️ 語音輸入";
    setMicState("mic_processing");
    try {
      const hadSignal = hasEnoughRecordedSignal();
      const blob = new Blob(mediaRecorderChunks, { type: mediaRecorder.mimeType || mimeType || "audio/webm" });
      if (!hadSignal) {
        cleanupRecorderStream();
        micHint.textContent = "錄音幾乎是靜音，沒有送出轉寫。請檢查麥克風輸入來源，或靠近麥克風再錄一次。";
        setMicState("mic_error");
        return;
      }
      const shouldAutoSubmit = Boolean(autoSubmitToggle.checked);
      if (shouldAutoSubmit) {
        // Plan 37 V3: one round trip — offline STT -> emotion -> Gemma -> TTS.
        setComposerBusy(true);
        setConversationState("awaiting_response");
        try {
          const voicePayload = await uploadRecordingForVoiceChat(blob);
          cleanupRecorderStream();
          appendMessage("user", voicePayload.transcript || "");
          const t = voicePayload.timings || {};
          micHint.textContent = `語音對話完成（STT ${t.stt_s ?? "-"}s / 總計 ${t.total_s ?? "-"}s）`;
          setMicState("mic_ready");
          deliverReply(voicePayload);
        } finally {
          setComposerBusy(false);
        }
        return;
      }
      const payload = await uploadRecordingForTranscription(blob);
      messageInput.value = payload.transcript || "";
      micHint.textContent = shouldAutoSubmit
        ? "本機轉寫完成，已自動送出"
        : "本機轉寫完成，等待你確認送出";
      cleanupRecorderStream();
      if (messageInput.value.trim() && shouldAutoSubmit) {
        await sendMessage(messageInput.value);
      } else {
        setMicState("mic_ready");
        setConversationState("ready");
      }
    } catch (error) {
      cleanupRecorderStream();
      micHint.textContent = `本機語音轉寫失敗：${formatMicError(error)}`;
      setMicState("mic_error");
    }
  };

  mediaRecorder.start();
}

function stopServerRecording() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") {
    return;
  }
  mediaRecorder.stop();
}

function stopSpeechPlayback() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (!("speechSynthesis" in window)) {
    return;
  }
  window.speechSynthesis.cancel();
  setSpeechState(speakToggle.checked ? "speech_ready" : "speech_disabled");
}

async function toggleCamera() {
  if (cameraActive) {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
    userVideo.srcObject = null;
    if (userVideoWrap) {
      userVideoWrap.classList.remove("active");
    }
    cameraActive = false;
    cameraToggle.textContent = "📷";
    cameraToggle.classList.remove("active");
    appendSystemMessage("視訊鏡頭已關閉");
  } else {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false
      });
      userVideo.srcObject = cameraStream;
      if (userVideoWrap) {
        userVideoWrap.classList.add("active");
      }
      cameraActive = true;
      cameraToggle.textContent = "🚫";
      cameraToggle.classList.add("active");
      appendSystemMessage("視訊鏡頭已開啟");
    } catch (error) {
      appendSystemMessage(`開啟鏡頭失敗：${formatMicError(error)}`);
    }
  }
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload.error || "healthcheck_failed");
    }
    voiceConfig = { ...voiceConfig, ...(payload.voice || {}) };
    avatarConfig = { ...avatarConfig, ...(payload.avatar || {}) };
    renderLife(payload.life);
    serverTtsAvailable = Boolean(payload.tts && payload.tts.ok);
    if (payload.dna && dnaStatus) {
      dnaStatus.textContent = `${payload.dna.genome_id} g${payload.dna.generation}`;
    }
    if (serviceStatus) {
      serviceStatus.textContent = "已連線";
    }
    backendStatus.textContent = `${payload.llm_healthcheck.backend} / ${payload.llm_healthcheck.model_family}`;
    speakToggle.checked = Boolean(voiceConfig.auto_speak);
    autoSubmitToggle.checked = Boolean(voiceConfig.auto_submit);
    setSpeechState(speakToggle.checked ? "speech_ready" : "speech_disabled");
    await queryMicrophonePermission();
    clearErrorState();
    configureRecognition();
  } catch (error) {
    if (serviceStatus) {
      serviceStatus.textContent = "連線失敗";
    }
    backendStatus.textContent = String(error);
    appendSystemMessage(`服務連線失敗：${error}`);
    setConversationState("failed");
  }
}

async function sendMessage(message) {
  const trimmed = message.trim();
  if (!trimmed) {
    return;
  }

  appendMessage("user", trimmed);
  messageInput.value = "";
  setComposerBusy(true);
  stopSpeechPlayback();
  clearErrorState();
  setConversationState("submitting");
  setConversationState("awaiting_response");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: trimmed, tts: Boolean(speakToggle.checked && serverTtsAvailable) }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
    }
    deliverReply(payload);
  } catch (error) {
    appendSystemMessage(`送出失敗：${error}`);
    setConversationState("failed");
  } finally {
    setComposerBusy(false);
    messageInput.focus();
  }
}

function speakText(text) {
  if (!("speechSynthesis" in window)) {
    setSpeechState("speech_disabled");
    setConversationState("ready");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = voiceConfig.synthesis_lang || "zh-TW";
  utterance.rate = lastVoiceModulation && lastVoiceModulation.rate ? lastVoiceModulation.rate : 1.0;
  utterance.pitch = lastVoiceModulation && lastVoiceModulation.pitch ? lastVoiceModulation.pitch : 1.0;
  utterance.onstart = () => {
    setConversationState("reply_received");
    setSpeechState("speech_playing");
  };
  utterance.onend = () => {
    setSpeechState(speakToggle.checked ? "speech_ready" : "speech_disabled");
    setConversationState("ready");
  };
  utterance.onerror = () => {
    setSpeechState("speech_error");
    setConversationState("ready");
  };
  window.speechSynthesis.speak(utterance);
}

function configureRecognition() {
  if (isServerRecordingMode()) {
    if (!window.isSecureContext) {
      micHint.textContent = "目前頁面不是安全環境，請改用 localhost 或 https";
      setMicState("mic_error");
      return;
    }
    if (!supportsServerRecording()) {
      micHint.textContent = "此瀏覽器不支援錄音上傳模式，請改用 Chrome 或 Edge";
      setMicState("mic_unavailable");
      return;
    }
    micHint.textContent = "可使用本機錄音轉寫，按一下開始錄音，再按一次停止";
    setMicState("mic_ready");
    return;
  }

  if (!SpeechRecognition) {
    micButton.disabled = true;
    micHint.textContent = "此瀏覽器不支援 Web Speech API，請改用 Chrome 或 Edge";
    setMicState("mic_unavailable");
    return;
  }

  if (!window.isSecureContext) {
    micHint.textContent = "目前不是安全頁面環境，麥克風可能無法使用";
  }

  recognition = new SpeechRecognition();
  recognition.lang = voiceConfig.recognition_lang || "zh-TW";
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  micHint.textContent = "可使用麥克風語音輸入，第一次使用會要求權限";
  setMicState("mic_ready");

  recognition.onstart = () => {
    recognitionActive = true;
    micButton.textContent = "⏹️ 停止收音";
    micHint.textContent = "正在聽你說話";
    stopSpeechPlayback();
    clearErrorState();
    setMicState("mic_listening");
  };

  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map((result) => result[0].transcript)
      .join("")
      .trim();
    messageInput.value = transcript;
  };

  recognition.onnomatch = () => {
    micHint.textContent = "沒有辨識到清楚語音，請再試一次";
  };

  recognition.onend = async () => {
    const finalText = messageInput.value.trim();
    recognitionActive = false;
    micButton.textContent = "🎙️ 語音輸入";
    setMicState("mic_processing");
    const shouldAutoSubmit = Boolean(autoSubmitToggle.checked);
    micHint.textContent = finalText
      ? shouldAutoSubmit
        ? "語音已轉成文字，已自動送出"
        : "語音已轉成文字，等待你確認送出"
      : "沒有辨識到語音";
    if (finalText && shouldAutoSubmit) {
      await sendMessage(finalText);
    } else {
      setMicState("mic_ready");
      setConversationState("ready");
    }
  };

  recognition.onerror = (event) => {
    recognitionActive = false;
    micButton.textContent = "🎙️ 語音輸入";
    micHint.textContent = `語音辨識失敗：${event.error}`;
    setMicState("mic_error");
  };
}

composerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage(messageInput.value);
});

micButton.addEventListener("click", async () => {
  clearErrorState();

  if (isServerRecordingMode()) {
    if (mediaRecorderActive) {
      stopServerRecording();
      return;
    }
    try {
      await startServerRecording();
    } catch (error) {
      cleanupRecorderStream();
      micHint.textContent = `啟動本機錄音失敗：${formatMicError(error)}`;
      setMicState("mic_error");
    }
    return;
  }

  if (!recognition) {
    micHint.textContent = "目前瀏覽器沒有可用的語音辨識功能";
    return;
  }

  if (recognitionActive) {
    try {
      recognition.stop();
    } catch (error) {
      micHint.textContent = `停止收音失敗：${formatMicError(error)}`;
      setMicState("mic_error");
    }
    return;
  }

  if (!window.isSecureContext) {
    micHint.textContent = "目前頁面不是安全環境，請改用 localhost 或 https";
    setMicState("mic_error");
    return;
  }

  const micAccess = await ensureMicrophoneAccess(true);
  if (!micAccess.ok) {
    micHint.textContent = `麥克風權限或裝置不可用：${micAccess.reason}`;
    setMicState("mic_error");
    return;
  }

  setMicState("mic_starting");
  messageInput.value = "";
  recognition.lang = voiceConfig.recognition_lang || recognition.lang;
  try {
    recognition.start();
  } catch (error) {
    micHint.textContent = `啟動語音輸入失敗：${formatMicError(error)}`;
    setMicState("mic_error");
  }
});

speakToggle.addEventListener("change", () => {
  voiceConfig.auto_speak = Boolean(speakToggle.checked);
  if (!speakToggle.checked) {
    stopSpeechPlayback();
  } else {
    setSpeechState("speech_ready");
  }
  syncAvatarState();
});

autoSubmitToggle.addEventListener("change", () => {
  voiceConfig.auto_submit = Boolean(autoSubmitToggle.checked);
});

if (resetButton) {
  resetButton.addEventListener("click", async () => {
    try {
      stopSpeechPlayback();
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      chatLog.innerHTML = "";
      clearErrorState();
      setMicState(isServerRecordingMode() ? "mic_ready" : recognition ? "mic_ready" : "mic_unavailable");
      setSpeechState(speakToggle.checked ? "speech_ready" : "speech_disabled");
      setConversationState("ready");
      appendSystemMessage("對話已重設，可以重新開始。")
    } catch (error) {
      appendSystemMessage(`重設失敗：${error}`);
      setConversationState("failed");
    }
  });
}

cameraToggle.addEventListener("click", toggleCamera);

for (const button of quickButtons) {
  button.addEventListener("click", async () => {
    const prompt = button.dataset.prompt || "";
    if (!prompt) {
      return;
    }
    await sendMessage(prompt);
  });
}

appendSystemMessage("Gemma 4 網頁對話已就緒。你可以打字，或按麥克風直接說話。");
fetchHealth();
syncAvatarState();
messageInput.focus();
