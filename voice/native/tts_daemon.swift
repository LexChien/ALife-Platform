// Plan 38 J2.4: resident macOS synthesizer (same engine and voice as `say -v Meijia`, no per-sentence process spawn).
// Protocol: one JSON object per stdin line {"id","text","voice","wpm","out"} -> one JSON line on stdout
// {"id","ok","out","synth_s","error"}. Output: 16-bit mono WAV at 22050 Hz (same as the `say` + afconvert path).
import AppKit
import AVFoundation
import Foundation

final class Job { let id: String; let text: String; let out: URL; let t0: Date
  init(id: String, text: String, out: URL) { self.id = id; self.text = text; self.out = out; self.t0 = Date() } }

final class Daemon: NSObject, NSSpeechSynthesizerDelegate {
  var synth: NSSpeechSynthesizer
  var voiceName: String
  var queue: [Job] = []
  var current: Job?
  var aiff: URL?

  init(voice: String) {
    voiceName = voice
    synth = NSSpeechSynthesizer(voice: Daemon.voiceId(voice)) ?? NSSpeechSynthesizer()
    super.init()
    synth.delegate = self
  }

  static func voiceId(_ name: String) -> NSSpeechSynthesizer.VoiceName? {
    for v in NSSpeechSynthesizer.availableVoices {
      let attrs = NSSpeechSynthesizer.attributes(forVoice: v)
      if let n = attrs[.name] as? String, n.lowercased() == name.lowercased() { return v }
    }
    return nil
  }

  func emit(_ obj: [String: Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: obj), let s = String(data: d, encoding: .utf8) {
      FileHandle.standardOutput.write((s + "\n").data(using: .utf8)!)
    }
  }

  func submit(_ o: [String: Any]) {
    let id = o["id"] as? String ?? UUID().uuidString
    guard let text = o["text"] as? String, let out = o["out"] as? String else {
      emit(["id": id, "ok": false, "error": "text and out required"]); return }
    if let v = o["voice"] as? String, v != voiceName, let vid = Daemon.voiceId(v) {
      synth.setVoice(vid); voiceName = v }
    if let wpm = o["wpm"] as? Double { synth.rate = Float(wpm) }
    queue.append(Job(id: id, text: text, out: URL(fileURLWithPath: out)))
    pump()
  }

  func pump() {
    guard current == nil, !queue.isEmpty else { return }
    let job = queue.removeFirst(); current = job
    let tmp = job.out.deletingPathExtension().appendingPathExtension("partial.aiff")
    aiff = tmp
    if !synth.startSpeaking(job.text, to: tmp) { finish(ok: false, error: "startSpeaking failed") }
  }

  func speechSynthesizer(_ sender: NSSpeechSynthesizer, didFinishSpeaking finishedSpeaking: Bool) {
    finish(ok: finishedSpeaking, error: finishedSpeaking ? nil : "interrupted")
  }

  func finish(ok: Bool, error: String?) {
    guard let job = current else { return }
    var okv = ok; var err = error
    if ok, let src = aiff {
      do {
        let inFile = try AVAudioFile(forReading: src)
        let fmt = inFile.processingFormat
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 22050.0,
          AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        let partial = job.out.deletingPathExtension().appendingPathExtension("partial.wav")
        try? FileManager.default.removeItem(at: partial)
        let outFile = try AVAudioFile(forWriting: partial, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let cap = AVAudioFrameCount(inFile.length)
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: max(cap, 1))!
        try inFile.read(into: buf)
        if fmt.sampleRate == 22050 && fmt.channelCount == 1 {
          try outFile.write(from: buf)
        } else {
          let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 22050, channels: 1, interleaved: false)!
          let conv = AVAudioConverter(from: fmt, to: target)!
          let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(cap) * 22050 / fmt.sampleRate) + 1024)!
          var fed = false
          var cerr: NSError?
          conv.convert(to: outBuf, error: &cerr) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true; status.pointee = .haveData; return buf }
          try outFile.write(from: outBuf)
        }
        try? FileManager.default.removeItem(at: job.out)
        try FileManager.default.moveItem(at: partial, to: job.out)
        try? FileManager.default.removeItem(at: src)
      } catch { okv = false; err = "convert: \(error)" }
    }
    emit(["id": job.id, "ok": okv, "out": job.out.path, "voice": voiceName,
          "synth_s": Date().timeIntervalSince(job.t0), "error": err ?? NSNull()])
    current = nil
    pump()
  }
}

let voice = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Meijia"
let daemon = Daemon(voice: voice)
daemon.emit(["ready": true, "voice": voice, "voice_found": Daemon.voiceId(voice) != nil, "pid": ProcessInfo.processInfo.processIdentifier])
DispatchQueue.global().async {
  while let line = readLine(strippingNewline: true) {
    guard let d = line.data(using: .utf8), let o = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else { continue }
    DispatchQueue.main.async { daemon.submit(o) }
  }
  DispatchQueue.main.async { exit(0) }
}
RunLoop.main.run()
