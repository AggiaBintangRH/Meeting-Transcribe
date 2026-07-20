// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MeetingTranscriber",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MeetingTranscriber",
            path: "Sources/MeetingTranscriber",
            resources: [
                .process("Assets.xcassets")
            ]
        ),
        .testTarget(
            name: "MeetingTranscriberTests",
            dependencies: ["MeetingTranscriber"],
            path: "Tests/MeetingTranscriberTests"
        )
    ]
)
