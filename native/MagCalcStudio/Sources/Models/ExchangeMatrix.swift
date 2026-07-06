import Foundation

/// Port of gui/src/lib/exchangeMatrix.js: turns an interaction (type + value)
/// into a displayable 3×3 exchange tensor, keeping parameter names symbolic
/// with the same cosmetic cleanups as the web app.
enum ExchangeMatrix {
    /// Symbolic 3×3 matrix (entries are display strings; "0" for zeros).
    /// Returns nil for types without a simple tensor form (interaction_matrix,
    /// kitaev), matching the web helper.
    static func symbolic(type: String, value: JSONValue?) -> [[String]]? {
        func vec3(_ v: JSONValue?) -> [String] {
            if let arr = v?.arrayValue {
                var out = arr.map { cleanSymbol($0) }
                while out.count < 3 { out.append("0") }
                return out
            }
            if let s = v?.stringValue, s.contains(",") {
                var out = s.split(separator: ",").map { cleanSymbol(.string(String($0))) }
                while out.count < 3 { out.append("0") }
                return out
            }
            return ["0", "0", "0"]
        }

        switch type {
        case "heisenberg":
            let j = cleanSymbol(value ?? .string("0"))
            return [[j, "0", "0"], ["0", j, "0"], ["0", "0", j]]
        case "dm", "dm_interaction", "dm_manual":
            let v = vec3(value)
            let (dx, dy, dz) = (v[0], v[1], v[2])
            return [["0", dz, neg(dy)], [neg(dz), "0", dx], [dy, neg(dx), "0"]]
        case "anisotropic", "anisotropic_exchange":
            let v = vec3(value)
            return [[v[0], "0", "0"], ["0", v[1], "0"], ["0", "0", v[2]]]
        default:
            return nil
        }
    }

    static func neg(_ v: String) -> String {
        if v == "0" { return "0" }
        if v.hasPrefix("-") { return String(v.dropFirst()) }
        return "-" + v
    }

    /// Mirrors getSymbol() in exchangeMatrix.js: rounds embedded floats to
    /// 5 decimals, zeroes tiny values, and strips 1.0*/0*-style noise.
    static func cleanSymbol(_ value: JSONValue) -> String {
        if let n = value.doubleValue, value.stringValue == nil || Double(value.stringValue!) != nil {
            if abs(n) < 1e-10 { return "0" }
            let rounded = (n * 100000).rounded() / 100000
            if rounded.truncatingRemainder(dividingBy: 1) == 0, abs(rounded) < 1e15 {
                return String(Int64(rounded))
            }
            return String(rounded)
        }

        var s = (value.stringValue ?? value.displayString).trimmingCharacters(in: .whitespaces)

        // Round decimal literals inside the expression to 5 places.
        if let regex = try? NSRegularExpression(pattern: #"[+-]?\d*\.\d+(?:[eE][+-]?\d+)?"#) {
            var result = ""
            var last = s.startIndex
            let ns = s as NSString
            for match in regex.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
                guard let range = Range(match.range, in: s) else { continue }
                result += s[last..<range.lowerBound]
                let token = String(s[range])
                if let f = Double(token) {
                    if abs(f) < 1e-10 {
                        result += "0"
                    } else {
                        let rounded = (f * 100000).rounded() / 100000
                        var str = rounded.truncatingRemainder(dividingBy: 1) == 0
                            ? String(Int64(rounded)) : String(rounded)
                        if str.hasPrefix(".") { str = "0" + str }
                        if str.hasPrefix("-.") { str = "-0" + str.dropFirst(1) }
                        result += str
                    }
                } else {
                    result += token
                }
                last = range.upperBound
            }
            result += s[last...]
            s = result
        }

        // Cosmetic algebraic simplifications (same regex set as the web app).
        let simplifications: [(String, String)] = [
            (#"\b0\s*\*\s*[a-zA-Z0-9_]+"#, "0"),
            (#"[a-zA-Z0-9_]+\s*\*\s*0\b"#, "0"),
            (#"\+\s*0\b"#, ""),
            (#"\b0\s*\+\s*"#, ""),
            (#"\-\s*0\b"#, ""),
            (#"\b0\s*\-\s*"#, "-"),
        ]
        for (pattern, replacement) in simplifications {
            s = s.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }

        if s.trimmingCharacters(in: .whitespaces).isEmpty || s == "-0" || s == "-0.0" { s = "0" }
        if s.hasPrefix("1.0*") { s = String(s.dropFirst(4)) }
        else if s.hasPrefix("1*") { s = String(s.dropFirst(2)) }
        if s.hasPrefix("-1.0*") { s = "-" + s.dropFirst(5) }
        else if s.hasPrefix("-1*") { s = "-" + s.dropFirst(3) }
        if s == "0" || s == "0.0" { return "0" }
        if let f = Double(s), abs(f) < 1e-10 { return "0" }
        return s
    }
}
