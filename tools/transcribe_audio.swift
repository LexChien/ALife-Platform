import Foundation
import Speech

func fail(_ code: Int32, _ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    fail(1, "usage: transcribe_audio.swift <audio-file> [locale]")
}

let audioPath = args[1]
let localeIdentifier = args.count >= 3 ? args[2] : "zh-TW"
let audioURL = URL(fileURLWithPath: audioPath)

guard FileManager.default.fileExists(atPath: audioPath) else {
    fail(2, "audio file not found: \(audioPath)")
}

SFSpeechRecognizer.requestAuthorization { status in
    switch status {
    case .authorized:
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
            ?? SFSpeechRecognizer() else {
            fail(3, "failed to create recognizer for locale: \(localeIdentifier)")
        }
        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        request.shouldReportPartialResults = false
        recognizer.recognitionTask(with: request) { result, error in
            if let error = error {
                fail(4, "speech error: \(error.localizedDescription)")
            }
            guard let result = result, result.isFinal else {
                return
            }
            print(result.bestTranscription.formattedString)
            exit(0)
        }
    case .denied:
        fail(5, "speech authorization denied")
    case .restricted:
        fail(6, "speech authorization restricted")
    case .notDetermined:
        fail(7, "speech authorization not determined")
    @unknown default:
        fail(8, "speech authorization unknown")
    }
}

RunLoop.main.run()
