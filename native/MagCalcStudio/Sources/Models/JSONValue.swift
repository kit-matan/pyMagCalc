import Foundation

/// A dynamically-typed JSON value. The MagCalc config format is heterogeneous
/// (interaction `value` can be a string like "J1", a 3-vector of expressions,
/// or a 3x3 symbolic matrix; parameters mix scalars and vectors), so the
/// native model uses this type wherever the web app stored free-form JSON.
enum JSONValue: Codable, Hashable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let n = try? container.decode(Double.self) {
            self = .number(n)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([JSONValue].self) {
            self = .array(a)
        } else if let o = try? container.decode([String: JSONValue].self) {
            self = .object(o)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let b): try container.encode(b)
        case .number(let n):
            // Encode integral numbers without a trailing ".0" where possible.
            if n.truncatingRemainder(dividingBy: 1) == 0, abs(n) < 1e15 {
                try container.encode(Int64(n))
            } else {
                try container.encode(n)
            }
        case .string(let s): try container.encode(s)
        case .array(let a): try container.encode(a)
        case .object(let o): try container.encode(o)
        }
    }

    // MARK: Convenience accessors

    var doubleValue: Double? {
        switch self {
        case .number(let n): return n
        case .string(let s): return Double(s)
        case .bool(let b): return b ? 1 : 0
        default: return nil
        }
    }

    var stringValue: String? {
        switch self {
        case .string(let s): return s
        case .number(let n):
            if n.truncatingRemainder(dividingBy: 1) == 0, abs(n) < 1e15 {
                return String(Int64(n))
            }
            return String(n)
        case .bool(let b): return b ? "true" : "false"
        default: return nil
        }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    var objectValue: [String: JSONValue]? {
        if case .object(let o) = self { return o }
        return nil
    }

    var isVector: Bool { arrayValue != nil }

    /// Display string for UI tables ("J1", "[Dx, 0, 0]", "3×3 matrix", 2.49, …).
    var displayString: String {
        switch self {
        case .null: return "—"
        case .bool(let b): return b ? "true" : "false"
        case .number: return stringValue ?? "0"
        case .string(let s): return s
        case .array(let a):
            if a.allSatisfy({ $0.arrayValue != nil }) && a.count == 3 {
                return "3×3 matrix"
            }
            return "[" + a.map { $0.displayString }.joined(separator: ", ") + "]"
        case .object: return "{…}"
        }
    }
}

extension JSONValue: ExpressibleByStringLiteral, ExpressibleByFloatLiteral,
                     ExpressibleByIntegerLiteral, ExpressibleByBooleanLiteral,
                     ExpressibleByArrayLiteral {
    init(stringLiteral value: String) { self = .string(value) }
    init(floatLiteral value: Double) { self = .number(value) }
    init(integerLiteral value: Int) { self = .number(Double(value)) }
    init(booleanLiteral value: Bool) { self = .bool(value) }
    init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
}
