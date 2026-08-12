import Foundation

enum APIError: LocalizedError {
    case badStatus(Int, String)
    case notConnected

    var errorDescription: String? {
        switch self {
        case .badStatus(let code, let detail):
            return detail.isEmpty ? "Server returned status \(code)" : detail
        case .notConnected:
            return "Not connected to a MagCalc backend."
        }
    }
}

/// Async client for the MagCalc FastAPI backend (gui/server.py).
struct APIClient: Sendable {
    var baseURL: URL

    private var session: URLSession { .shared }

    // MARK: Generic plumbing

    private func postJSON<T: Decodable>(_ path: String, body: JSONValue,
                                        as type: T.Type,
                                        timeout: TimeInterval = 60) async throws -> T {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        request.timeoutInterval = timeout
        let (data, response) = try await session.data(for: request)
        try Self.checkStatus(response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private static func checkStatus(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            var detail = ""
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let d = obj["detail"] as? String {
                detail = d
            }
            throw APIError.badStatus(http.statusCode, detail)
        }
    }

    private func postMultipart<T: Decodable>(_ path: String, fileURL: URL,
                                             as type: T.Type) async throws -> T {
        let boundary = "magcalc-\(UUID().uuidString)"
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let needsAccess = fileURL.startAccessingSecurityScopedResource()
        defer { if needsAccess { fileURL.stopAccessingSecurityScopedResource() } }
        let fileData = try Data(contentsOf: fileURL)

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        request.timeoutInterval = 120

        let (data, response) = try await session.data(for: request)
        try Self.checkStatus(response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    // MARK: Endpoints

    func health() async -> Bool {
        var request = URLRequest(url: baseURL.appendingPathComponent("docs"))
        request.httpMethod = "HEAD"
        request.timeoutInterval = 3
        guard let (_, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse else { return false }
        return http.statusCode < 500
    }

    func parseCIF(fileURL: URL) async throws -> ParsedCIF {
        try await postMultipart("parse-cif", fileURL: fileURL, as: ParsedCIF.self)
    }

    func parseMCIF(fileURL: URL) async throws -> ParsedMCIF {
        try await postMultipart("parse-mcif", fileURL: fileURL, as: ParsedMCIF.self)
    }

    func uploadFitData(fileURL: URL) async throws -> FitDataUploadResult {
        try await postMultipart("upload-fit-data", fileURL: fileURL, as: FitDataUploadResult.self)
    }

    func neighbors(for config: MagCalcConfig, maxDistance: Double = 10.0) async throws -> [NeighborShell] {
        try await postJSON("get-neighbors",
                           body: .object(["data": config.structurePayload(),
                                          "max_distance": .number(maxDistance)]),
                           as: [NeighborShell].self)
    }

    func visualizerData(for config: MagCalcConfig) async throws -> VisualizerData {
        try await postJSON("get-visualizer-data",
                           body: .object(["data": config.structurePayload(includeInteractions: true)]),
                           as: VisualizerData.self)
    }

    func analyzeBonds(for config: MagCalcConfig, maxDistance: Double = 10.0) async throws -> [BondOrbit] {
        try await postJSON("analyze-bonds",
                           body: .object(["data": config.structurePayload(),
                                          "max_distance": .number(maxDistance)]),
                           as: [BondOrbit].self)
    }

    func bondConstraints(for config: MagCalcConfig, bond: BondOrbit.Representative) async throws -> BondConstraints {
        let bondValue: JSONValue = .object([
            "atom_i": bond.atomI,
            "atom_j": bond.atomJ,
            "offset": .array(bond.offset.map { .number(Double($0)) }),
        ])
        return try await postJSON("bond-constraints",
                                  body: .object(["data": config.structurePayload(),
                                                 "bond": bondValue]),
                                  as: BondConstraints.self)
    }

    func expandConfig(_ config: MagCalcConfig) async throws -> JSONValue {
        try await postJSON("expand-config",
                           body: .object(["data": config.backendInput()]),
                           as: JSONValue.self)
    }

    /// Long-running: returns when the calculation subprocess finishes.
    ///
    /// The run endpoint is "config as source": it takes the config under
    /// `config` (NOT the `data` envelope the structure endpoints use), writes it
    /// verbatim and hands the file to the same `magcalc.runner.run_calculation`
    /// the CLI uses. This client was still sending `data`, so the server answered
    /// 400 "Payload must contain 'config' or 'path'" for every run.
    ///
    /// `config_dir` is the directory of the file the user opened, so relative
    /// references in it (`from_mcif`, `fitting.data_file`, `cif_file`) resolve
    /// exactly as they do for `magcalc run <that file>`.
    func runCalculation(_ config: MagCalcConfig,
                        taskOverrides: [String: JSONValue]? = nil,
                        configDir: String? = nil) async throws -> CalculationResults {
        var body: [String: JSONValue] = [
            "config": config.backendInput(taskOverrides: taskOverrides),
        ]
        if let configDir, !configDir.isEmpty { body["config_dir"] = .string(configDir) }
        return try await postJSON("run-calculation",
                                  body: .object(body),
                                  as: CalculationResults.self,
                                  timeout: 3600)
    }

    func stopCalculation() async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("stop-calculation"))
        request.httpMethod = "POST"
        let (data, response) = try await session.data(for: request)
        try Self.checkStatus(response, data: data)
    }

    /// Fetch a server file (plot image, mag_structure.json) with cache busting.
    ///
    /// Result paths come back with the web app's `/api` prefix, which exists only
    /// for the browser's Vite dev proxy (it rewrites `/api/x` -> `/x` on :8000).
    /// This client talks to the server directly, so the prefix has to come off or
    /// every plot 404s.
    func fileURL(_ serverPath: String) -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.path = serverPath.hasPrefix("/api/")
            ? String(serverPath.dropFirst("/api".count))
            : serverPath
        components.queryItems = [URLQueryItem(name: "t", value: String(Int(Date().timeIntervalSince1970 * 1000)))]
        return components.url!
    }

    func fetchFile(_ serverPath: String) async throws -> Data {
        let (data, response) = try await session.data(from: fileURL(serverPath))
        try Self.checkStatus(response, data: data)
        return data
    }

    /// Run outputs are served from the LAST RUN's directory via /run-artifact,
    /// not from the project-root /files mount -- a run started from an opened
    /// file's own directory writes its plots there.
    func fetchMagStructure() async throws -> MagStructureResult {
        let data = try await fetchFile("/run-artifact/mag_structure.json")
        return try JSONDecoder().decode(MagStructureResult.self, from: data)
    }
}
