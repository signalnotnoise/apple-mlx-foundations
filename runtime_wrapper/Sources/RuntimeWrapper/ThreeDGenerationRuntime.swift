import CoreGraphics
import Foundation
import ShapERuntime

public enum ThreeDGenerationModel: String, CaseIterable, Codable, Sendable {
    case shapE = "shap-e"
    case tripoSR = "triposr"
}

public enum ThreeDGenerationConfiguration: Sendable {
    case shapE(modelBundleURL: URL, bpeVocabularyURL: URL)
    case tripoSR(weightsURL: URL)

    public var model: ThreeDGenerationModel {
        switch self {
        case .shapE: return .shapE
        case .tripoSR: return .tripoSR
        }
    }
}

public enum ThreeDGenerationRuntimeError: Error, LocalizedError, Equatable {
    case promptUnsupported(ThreeDGenerationModel)

    public var errorDescription: String? {
        switch self {
        case .promptUnsupported(let model):
            return "The \(model.rawValue) model does not support text prompts. Use an image input."
        }
    }
}

/// Model-selecting facade over the independent Shap-E and TripoSR packages.
public final class ThreeDGenerationRuntime: @unchecked Sendable {
    public typealias ProgressCallback = ShapERuntime.ShapEProgressCallback

    private enum Backend {
        case shapE(ShapERuntime)
        case tripoSR(TripoSRMeshRuntime)
    }

    public let selectedModel: ThreeDGenerationModel
    private let backend: Backend

    public init(configuration: ThreeDGenerationConfiguration) throws {
        self.selectedModel = configuration.model
        switch configuration {
        case .shapE(let modelBundleURL, let bpeVocabularyURL):
            self.backend = .shapE(
                try ShapERuntime(
                    modelBundleURL: modelBundleURL,
                    bpeVocabularyURL: bpeVocabularyURL
                )
            )
        case .tripoSR(let weightsURL):
            self.backend = .tripoSR(try TripoSRMeshRuntime(weightsURL: weightsURL))
        }
    }

    public var shapERuntime: ShapERuntime? {
        guard case .shapE(let runtime) = backend else { return nil }
        return runtime
    }

    public var tripoSRRuntime: TripoSRMeshRuntime? {
        guard case .tripoSR(let runtime) = backend else { return nil }
        return runtime
    }

    /// Generates meshes from an image with model-appropriate defaults.
    public func generate(
        image: CGImage,
        gridSize: Int? = nil,
        queryBatchSize: Int? = nil,
        useMetalMarchingCubes: Bool = false,
        progress: ProgressCallback? = nil
    ) async throws -> [ThreeDMeshResult] {
        switch backend {
        case .shapE(let runtime):
            return try await runtime.generate(
                image: image,
                batchSize: 1,
                gridSize: gridSize ?? 128,
                queryBatchSize: queryBatchSize ?? 65_536,
                useMetalMarchingCubes: useMetalMarchingCubes,
                progress: progress
            ).map { ThreeDMeshResult(meshResult: $0.meshResult) }
        case .tripoSR(let runtime):
            let options = TripoSRMeshOptions(
                gridSize: gridSize ?? 256,
                queryBatchSize: queryBatchSize ?? 8_192,
                useMetalMarchingCubes: useMetalMarchingCubes
            )
            return try runtime.generate(
                image: image,
                options: options,
                progress: progress.map { callback in
                    { phase, step, total in callback(phase, step, total, nil) }
                }
            )
        }
    }

    /// Text generation is available only when the selected backend is Shap-E.
    public func generate(
        prompt: String,
        gridSize: Int = 128,
        queryBatchSize: Int = 65_536,
        useMetalMarchingCubes: Bool = false,
        progress: ProgressCallback? = nil
    ) async throws -> [ThreeDMeshResult] {
        guard case .shapE(let runtime) = backend else {
            throw ThreeDGenerationRuntimeError.promptUnsupported(selectedModel)
        }
        return try await runtime.generate(
            prompt: prompt,
            batchSize: 1,
            gridSize: gridSize,
            queryBatchSize: queryBatchSize,
            useMetalMarchingCubes: useMetalMarchingCubes,
            progress: progress
        ).map { ThreeDMeshResult(meshResult: $0.meshResult) }
    }
}
