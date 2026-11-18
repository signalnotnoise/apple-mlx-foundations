import Foundation
import MLX
import ShapERuntime

/// Model-neutral mesh result returned by `RuntimeWrapper`.
public struct ThreeDMeshResult: Sendable {
    public let meshResult: MeshResult

    public init(meshResult: MeshResult) {
        self.meshResult = meshResult
    }

    public func mlxVertices() throws -> MLXArray {
        try meshResult.vertices.asMLX().asType(Float.self)
    }

    public func mlxFaces() throws -> MLXArray {
        try meshResult.faces.asMLX().asType(Int32.self)
    }

    public func mlxTextureChannel(_ name: String) throws -> MLXArray? {
        try meshResult.vertexChannels?[name]?.asMLX().asType(Float.self)
    }

    public func verticesArray() throws -> [Float] {
        try mlxVertices().asArray(Float.self)
    }

    public func facesArray() throws -> [Int32] {
        try mlxFaces().asArray(Int32.self)
    }

    public func colorsArray() throws -> [Float]? {
        try mlxTextureChannel("RGBA")?.asArray(Float.self)
    }
}
