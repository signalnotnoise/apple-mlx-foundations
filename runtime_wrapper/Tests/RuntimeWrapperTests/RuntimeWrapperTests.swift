import MLX
import XCTest
@testable import RuntimeWrapper

final class RuntimeWrapperTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        _ = MLX.Stream.defaultStream(.cpu)
    }

    func testModelConfigurationReportsSelectedBackend() {
        let root = URL(fileURLWithPath: "/tmp")
        XCTAssertEqual(
            ThreeDGenerationConfiguration.shapE(
                modelBundleURL: root,
                bpeVocabularyURL: root
            ).model,
            .shapE
        )
        XCTAssertEqual(
            ThreeDGenerationConfiguration.tripoSR(weightsURL: root).model,
            .tripoSR
        )
        XCTAssertEqual(ThreeDGenerationModel.allCases, [.shapE, .tripoSR])
    }

    func testMeshOptionsRejectInvalidValues() {
        XCTAssertThrowsError(try TripoSRMeshOptions(gridSize: 1).validate()) { error in
            XCTAssertEqual(error as? TripoSRMeshRuntimeError, .invalidGridSize(1))
        }
        XCTAssertThrowsError(try TripoSRMeshOptions(queryBatchSize: 0).validate()) { error in
            XCTAssertEqual(error as? TripoSRMeshRuntimeError, .invalidQueryBatchSize(0))
        }
        XCTAssertThrowsError(try TripoSRMeshOptions(densityThreshold: .infinity).validate()) { error in
            XCTAssertEqual(
                error as? TripoSRMeshRuntimeError,
                .invalidDensityThreshold(.infinity)
            )
        }
    }

    func testQueryPointsUsesZThenYThenXOrdering() {
        let points = TripoSRMeshRuntime.queryPoints(
            start: 0,
            end: 8,
            gridSize: 2,
            radius: 1
        )
        XCTAssertEqual(points.shape, [8, 3])
        XCTAssertEqual(
            points.asArray(Float.self),
            [
                -1, -1, -1,
                -1, -1,  1,
                -1,  1, -1,
                -1,  1,  1,
                 1, -1, -1,
                 1, -1,  1,
                 1,  1, -1,
                 1,  1,  1,
            ]
        )
    }

    func testQueryPointChunksPreserveGlobalCoordinates() {
        let points = TripoSRMeshRuntime.queryPoints(
            start: 2,
            end: 6,
            gridSize: 2,
            radius: 1
        )
        XCTAssertEqual(
            points.asArray(Float.self),
            [
                -1,  1, -1,
                -1,  1,  1,
                 1, -1, -1,
                 1, -1,  1,
            ]
        )
    }
}
