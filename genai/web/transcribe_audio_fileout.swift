import Foundation
import Speech

func fail(_ code: Int32, _ message: String, _ outputPath: String?) -> Never {
    if let outputPath {
        try? ("ERROR: " + message + "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
    }
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    fail(1, "usage: ALifeTranscribeAudio <audio-file> <output-file> [locale]", nil)
}

let audioPath = args[1]
let outputPath = args[2]
let localeIdentifier = args.count >= 4 ? args[3] : "zh-TW"
let audioURL = URL(fileURLWithPath: audioPath)

guard FileManager.default.fileExists(atPath: audioPath) else {
    fail(2, "audio file not found: \(audioPath)", outputPath)
}

SFSpeechRecognizer.requestAuthorization { status in
    switch status {
    case .authorized:
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
            ?? SFSpeechRecognizer() else {
            fail(3, "failed to create recognizer for locale: \(localeIdentifier)", outputPath)
        }
        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        request.shouldReportPartialResults = false
        recognizer.recognitionTask(with: request) { result, error in
            if let error {
                fail(4, "speech error: \(error.localizedDescription)", outputPath)
            }
            guard let result, result.isFinal else {
                return
            }
            try? (result.bestTranscription.formattedString + "\n").write(
                toFile: outputPath,
                atomically: true,
                encoding: .utf8
            )
            exit(0)
        }
    case .denied:
        fail(5, "speech authorization denied", outputPath)
    case .restricted:
        fail(6, "speech authorization restricted", outputPath)
    case .notDetermined:
        fail(7, "speech authorization not determined", outputPath)
    @unknown default:
        fail(8, "speech authorization unknown", outputPath)
    }
}

RunLoop.main.run()
