import Foundation
@testable import ShapERuntime

enum RuntimeBundleFixture {
    static var bundleURL: URL {
        #if SWIFT_PACKAGE
        guard let url = Bundle.module.url(forResource: "runtime_bundle_stub", withExtension: nil, subdirectory: "Fixtures") else {
            fatalError("Missing checked-in runtime bundle fixture in Bundle.module")
        }
        return url
        #else
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let url = repoRoot.appendingPathComponent(
            "shap_e/swift/Tests/ShapERuntimeTests/Fixtures/runtime_bundle_stub",
            isDirectory: true
        )
        precondition(FileManager.default.fileExists(atPath: url.path), "Missing checked-in runtime bundle fixture at \(url.path)")
        return url
        #endif
    }
}
