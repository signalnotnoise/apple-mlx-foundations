import CoreGraphics
import Foundation
import MLX
import ShapERuntime
import TripoSRRuntime

public struct TripoSRMeshOptions: Sendable, Equatable {
    public var gridSize: Int
    public var queryBatchSize: Int
    public var densityThreshold: Float
    public var includeVertexColors: Bool
    public var useMetalMarchingCubes: Bool

    public init(
        gridSize: Int = 256,
        queryBatchSize: Int = 8_192,
        densityThreshold: Float = 25.0,
        includeVertexColors: Bool = true,
        useMetalMarchingCubes: Bool = false
    ) {
        self.gridSize = gridSize
        self.queryBatchSize = queryBatchSize
        self.densityThreshold = densityThreshold
        self.includeVertexColors = includeVertexColors
        self.useMetalMarchingCubes = useMetalMarchingCubes
    }

    public func validate() throws {
        guard gridSize >= 2 else { throw TripoSRMeshRuntimeError.invalidGridSize(gridSize) }
        guard queryBatchSize > 0 else {
            throw TripoSRMeshRuntimeError.invalidQueryBatchSize(queryBatchSize)
        }
        guard densityThreshold.isFinite, densityThreshold > 0 else {
            throw TripoSRMeshRuntimeError.invalidDensityThreshold(densityThreshold)
        }
    }
}

public enum TripoSRMeshRuntimeError: Error, LocalizedError, Equatable {
    case invalidGridSize(Int)
    case invalidQueryBatchSize(Int)
    case invalidDensityThreshold(Float)
    case invalidImageShape([Int])
    case gridSizeTooLarge(Int)
    case imageContextCreationFailed

    public var errorDescription: String? {
        switch self {
        case .invalidGridSize(let size):
            return "TripoSR gridSize must be at least 2, got \(size)."
        case .invalidQueryBatchSize(let size):
            return "TripoSR queryBatchSize must be greater than zero, got \(size)."
        case .invalidDensityThreshold(let threshold):
            return "TripoSR densityThreshold must be finite and greater than zero, got \(threshold)."
        case .invalidImageShape(let shape):
            return "TripoSR image must have shape [B,H,W,3], got \(shape)."
        case .gridSizeTooLarge(let size):
            return "TripoSR gridSize is too large to index safely: \(size)."
        case .imageContextCreationFailed:
            return "Failed to create the TripoSR image preprocessing context."
        }
    }
}

/// Adapts TripoSR density/color output to ShapERuntime's shared mesh result types.
public final class TripoSRMeshRuntime: @unchecked Sendable {
    public typealias ProgressCallback = (_ phase: String, _ step: Int, _ totalSteps: Int) -> Void

    public let model: TripoSRModel

    public init(weightsURL: URL) throws {
        let model = TripoSRModel()
        try model.loadWeights(from: weightsURL)
        MLX.eval(model.parameters())
        self.model = model
    }

    /// Converts a CGImage to the raw `[1,512,512,3]` RGB input expected by TripoSR.
    public static func preprocessImage(_ image: CGImage) throws -> MLXArray {
        let targetSize = 512
        let bytesPerPixel = 4
        let bytesPerRow = targetSize * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: targetSize * bytesPerRow)
        guard let context = CGContext(
            data: &pixelData,
            width: targetSize,
            height: targetSize,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw TripoSRMeshRuntimeError.imageContextCreationFailed
        }

        let gray = CGFloat(128) / 255
        context.setFillColor(red: gray, green: gray, blue: gray, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
        context.translateBy(x: 0, y: CGFloat(targetSize))
        context.scaleBy(x: 1, y: -1)
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))

        let rgba = MLXArray(pixelData, [1, targetSize, targetSize, 4])
        return rgba[0..., 0..., 0..., 0..<3].asType(.float32) / 255
    }

    public func generate(
        image: CGImage,
        options: TripoSRMeshOptions = TripoSRMeshOptions(),
        progress: ProgressCallback? = nil
    ) throws -> [ThreeDMeshResult] {
        try generate(image: Self.preprocessImage(image), options: options, progress: progress)
    }

    public func generate(
        image: MLXArray,
        options: TripoSRMeshOptions = TripoSRMeshOptions(),
        progress: ProgressCallback? = nil
    ) throws -> [ThreeDMeshResult] {
        try options.validate()
        guard image.ndim == 4, image.shape.last == 3, image.shape[0] > 0 else {
            throw TripoSRMeshRuntimeError.invalidImageShape(image.shape)
        }

        progress?("encoding", 0, 1)
        let triplanes = model.forwardTriplanes(image: image.asType(.float32))
        MLX.eval(triplanes)
        progress?("encoding", 1, 1)

        var results: [ThreeDMeshResult] = []
        results.reserveCapacity(image.shape[0])
        for batchIndex in 0..<image.shape[0] {
            let scene = triplanes[batchIndex..<(batchIndex + 1)]
            results.append(
                ThreeDMeshResult(
                    meshResult: try extractMesh(
                        triplanes: scene,
                        options: options,
                        progress: progress
                    )
                )
            )
        }
        return results
    }

    private func extractMesh(
        triplanes: MLXArray,
        options: TripoSRMeshOptions,
        progress: ProgressCallback?
    ) throws -> MeshResult {
        let gridSize = options.gridSize
        let (gSquared, squareOverflow) = gridSize.multipliedReportingOverflow(by: gridSize)
        let (total, cubeOverflow) = gSquared.multipliedReportingOverflow(by: gridSize)
        guard !squareOverflow, !cubeOverflow, total <= Int(Int32.max) else {
            throw TripoSRMeshRuntimeError.gridSizeTooLarge(gridSize)
        }

        let chunkCount = max(1, Int(ceil(Double(total) / Double(options.queryBatchSize))))
        var densityChunks: [MLXArray] = []
        densityChunks.reserveCapacity(chunkCount)

        var start = 0
        var chunkIndex = 0
        while start < total {
            let end = min(start + options.queryBatchSize, total)
            let positions = Self.queryPoints(
                start: start,
                end: end,
                gridSize: gridSize,
                radius: model.renderer.radius
            )
            let density = model.evaluate(triplanes: triplanes, positions: positions)
                .density
                .reshaped([-1])
                .asType(.float32)
            MLX.eval(density)
            densityChunks.append(density)
            start = end
            chunkIndex += 1
            progress?("volume", chunkIndex, chunkCount)
        }

        let field = MLX.concatenated(densityChunks, axis: 0)
            .reshaped([gridSize, gridSize, gridSize]) - options.densityThreshold
        MLX.eval(field)

        let fieldMin = MLX.min(field).item(Float.self)
        let fieldMax = MLX.max(field).item(Float.self)
        let radius = model.renderer.radius
        let bboxMin = MLXArray([-radius, -radius, -radius])
        let bboxSize = MLXArray([2 * radius, 2 * radius, 2 * radius])
        let bbox = BoundingBox(
            minCorner: [-radius, -radius, -radius],
            maxCorner: [radius, radius, radius]
        )
        let artifactInfo = ArtifactVersionInfo(artifactVersion: 1, manifestPath: "triposr")

        guard fieldMin <= 0, fieldMax >= 0 else {
            return try Self.emptyMesh(bbox: bbox, artifactInfo: artifactInfo)
        }

        progress?("meshing", 0, 1)
        let extracted: (verts: MLXArray, faces: MLXArray)
        if options.useMetalMarchingCubes {
            extracted = try MetalMarchingCubes.extract(
                sdf: field,
                bboxMin: bboxMin,
                bboxSize: bboxSize
            )
        } else {
            extracted = MarchingCubes.extract(
                sdf: field,
                bboxMin: bboxMin,
                bboxSize: bboxSize
            )
        }
        progress?("meshing", 1, 1)

        guard extracted.faces.shape[0] > 0 else {
            return try Self.emptyMesh(bbox: bbox, artifactInfo: artifactInfo)
        }

        let vertexChannels = try options.includeVertexColors
            ? evaluateVertexColors(
                triplanes: triplanes,
                vertices: extracted.verts,
                queryBatchSize: options.queryBatchSize,
                progress: progress
            )
            : nil

        let vertexValues = extracted.verts.asType(.float32).asArray(Float.self)
        let faceValues = extracted.faces.asType(.int32).asArray(Int32.self)
        let vertices = try RuntimeTensor(
            name: "verts",
            shape: extracted.verts.shape,
            dataType: .float32,
            data: vertexValues.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        let faces = try RuntimeTensor(
            name: "faces",
            shape: extracted.faces.shape,
            dataType: .int32,
            data: faceValues.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        return MeshResult(
            vertices: vertices,
            faces: faces,
            vertexChannels: vertexChannels,
            bbox: bbox,
            timings: [:],
            artifactInfo: artifactInfo
        )
    }

    private func evaluateVertexColors(
        triplanes: MLXArray,
        vertices: MLXArray,
        queryBatchSize: Int,
        progress: ProgressCallback?
    ) throws -> [String: RuntimeTensor] {
        let vertexCount = vertices.shape[0]
        let chunkCount = max(1, Int(ceil(Double(vertexCount) / Double(queryBatchSize))))
        var chunks: [MLXArray] = []
        chunks.reserveCapacity(chunkCount)

        var start = 0
        var chunkIndex = 0
        while start < vertexCount {
            let end = min(start + queryBatchSize, vertexCount)
            let color = model.evaluate(
                triplanes: triplanes,
                positions: vertices[start..<end]
            ).color.squeezed(axes: [0]).asType(.float32)
            MLX.eval(color)
            chunks.append(color)
            start = end
            chunkIndex += 1
            progress?("colors", chunkIndex, chunkCount)
        }

        let rgb = MLX.concatenated(chunks, axis: 0)
        let alpha = MLX.ones([vertexCount, 1], type: Float.self)
        let rgba = MLX.concatenated([rgb, alpha], axis: 1).flattened().asType(.float32)
        let values = rgba.asArray(Float.self)
        let tensor = try RuntimeTensor(
            name: "RGBA",
            shape: [values.count],
            dataType: .float32,
            data: values.withUnsafeBufferPointer { Data(buffer: $0) }
        )
        return ["RGBA": tensor]
    }

    static func queryPoints(
        start: Int,
        end: Int,
        gridSize: Int,
        radius: Float
    ) -> MLXArray {
        let gSquared = gridSize * gridSize
        let indices = MLX.arange(start, end, dtype: .int32)
        let zs = (indices % gridSize).asType(.float32)
        let ys = (indices.floorDivide(gridSize) % gridSize).asType(.float32)
        let xs = indices.floorDivide(gSquared).asType(.float32)
        let grid = MLX.stacked([xs, ys, zs], axis: 1)
        return (grid / Float(gridSize - 1)) * (2 * radius) - radius
    }

    private static func emptyMesh(
        bbox: BoundingBox,
        artifactInfo: ArtifactVersionInfo
    ) throws -> MeshResult {
        let vertices = try RuntimeTensor(
            name: "verts", shape: [0, 3], dataType: .float32, data: Data()
        )
        let faces = try RuntimeTensor(
            name: "faces", shape: [0, 3], dataType: .int32, data: Data()
        )
        return MeshResult(
            vertices: vertices,
            faces: faces,
            vertexChannels: nil,
            bbox: bbox,
            timings: [:],
            artifactInfo: artifactInfo
        )
    }
}
