// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "RuntimeWrapper",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "RuntimeWrapper",
            targets: ["RuntimeWrapper"]
        ),
    ],
    dependencies: [
        .package(name: "ShapERuntime", path: "../shap_e/swift"),
        .package(name: "TripoSR", path: "../TripoSR"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
    ],
    targets: [
        .target(
            name: "RuntimeWrapper",
            dependencies: [
                .product(name: "ShapERuntime", package: "ShapERuntime"),
                .product(name: "TripoSRRuntime", package: "TripoSR"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "RuntimeWrapperTests",
            dependencies: [
                "RuntimeWrapper",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
