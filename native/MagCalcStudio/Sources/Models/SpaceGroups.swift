import Foundation

struct SpaceGroup: Codable, Identifiable, Hashable {
    var number: Int
    var symbol: String

    var id: Int { number }
    var display: String { "\(number) · \(symbol)" }
}

enum SpaceGroups {
    static let all: [SpaceGroup] = {
        guard let url = Bundle.main.url(forResource: "space_groups", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let groups = try? JSONDecoder().decode([SpaceGroup].self, from: data) else {
            return (1...230).map { SpaceGroup(number: $0, symbol: "SG \($0)") }
        }
        return groups
    }()

    static func symbol(for number: Int) -> String {
        all.first { $0.number == number }?.symbol ?? "?"
    }
}
