// Plan 38 J2.1: AudioWorklet that downsamples the mic to 16 kHz mono Int16 and posts 20 ms (320-sample) frames.
class PcmUploader extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.out = new Int16Array(320);
    this.n = 0;
    this.sum = 0;
    this.cnt = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sum += ch[i];
      this.cnt += 1;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.sum / this.cnt));
        this.sum = 0;
        this.cnt = 0;
        this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.n === 320) {
          this.port.postMessage(this.out.buffer.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-uploader", PcmUploader);
