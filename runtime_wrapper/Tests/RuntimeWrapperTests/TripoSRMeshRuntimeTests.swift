import CoreGraphics
import Foundation
import ImageIO
import ShapERuntime
import XCTest

@testable import RuntimeWrapper

/// End-to-end TripoSR mesh evidence using the real converted weights.
///
/// The test is skipped unless both environment variables point at existing
/// files, because the weights are a gitignored derived artifact:
///
/// - `TRIPOSR_WEIGHTS_PATH`: `triposr_swift_ready.safetensors`
/// - `TRIPOSR_IMAGE_PATH`: an input image (for example `TripoSR/examples/chair.png`)
///
/// A skip is not a pass: release evidence must run this test with the
/// variables set. Numerical Python↔Swift agreement of the underlying field
/// is gated by `TripoSR/scripts/run_parity.sh`; this test covers the mesh
/// stage that gate does not reach (grid sampling, marching cubes, vertex
/// colors, and topology sanity of the extracted surface).
final class TripoSRMeshRuntimeTests: XCTestCase {

    private func requiredFileURL(environmentKey: String) throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[environmentKey],
            !path.isEmpty,
            FileManager.default.fileExists(atPath: path)
        else {
            throw XCTSkip(
                "Set \(environmentKey) to an existing file to run the TripoSR mesh test.")
        }
        return URL(fileURLWithPath: path)
    }

    private func loadCGImage(url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            XCTFail("Could not decode image at \(url.path)")
            throw XCTSkip("Undecodable test image")
        }
        return image
    }

    func testRealWeightsProduceFiniteColoredMesh() throws {
        let weightsURL = try requiredFileURL(environmentKey: "TRIPOSR_WEIGHTS_PATH")
        let imageURL = try requiredFileURL(environmentKey: "TRIPOSR_IMAGE_PATH")
        let image = try loadCGImage(url: imageURL)

        let runtime = try TripoSRMeshRuntime(weightsURL: weightsURL)
        let options = TripoSRMeshOptions(gridSize: 64, queryBatchSize: 8_192)
        let results = try runtime.generate(image: image, options: options)

        XCTAssertEqual(results.count, 1)
        guard let mesh = results.first else { return }

        let vertices = try mesh.verticesArray()
        let faces = try mesh.facesArray()
        XCTAssertFalse(vertices.isEmpty, "mesh must contain vertices")
        XCTAssertFalse(faces.isEmpty, "mesh must contain faces")
        XCTAssertEqual(vertices.count % 3, 0)
        XCTAssertEqual(faces.count % 3, 0)

        // TripoSRRenderer queries inside radius 0.87; marching-cubes vertices
        // are interpolated on that grid and must stay in the bounding box.
        let radius: Float = 0.87
        XCTAssertTrue(vertices.allSatisfy { $0.isFinite && abs($0) <= radius + 1e-5 })

        let stats = MeshTopology.stats(
            vertexCount: vertices.count / 3, faces: faces)
        print("TripoSR mesh smoke: \(stats.summaryLine)")
        XCTAssertEqual(stats.invalidFaceCount, 0)
        XCTAssertEqual(stats.nonManifoldEdgeCount, 0)

        // Vertex colors are flattened RGBA float32, one RGBA per vertex,
        // with sigmoid RGB in (0, 1) and alpha pinned to 1.
        let colors = try XCTUnwrap(try mesh.colorsArray())
        XCTAssertEqual(colors.count, (vertices.count / 3) * 4)
        for vertex in 0..<(colors.count / 4) {
            let base = vertex * 4
            for channel in 0..<3 {
                XCTAssertTrue(colors[base + channel] >= 0 && colors[base + channel] <= 1)
            }
            XCTAssertEqual(colors[base + 3], 1)
        }
    }
}
