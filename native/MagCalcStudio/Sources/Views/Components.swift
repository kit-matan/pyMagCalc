import SwiftUI

// MARK: - Shared form controls

/// Labeled numeric text field that commits parsed values (locale-independent,
/// accepts "." decimal separator like the web app).
struct NumberField: View {
    let label: String
    @Binding var value: Double
    var format: String = "%g"

    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            if !label.isEmpty {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            TextField(label, text: $text)
                .textFieldStyle(.roundedBorder)
                #if os(iOS)
                .keyboardType(.numbersAndPunctuation)
                .autocorrectionDisabled()
                #endif
                .focused($focused)
                .onAppear { text = String(format: format, value) }
                .onChange(of: value) { _, newValue in
                    if !focused { text = String(format: format, newValue) }
                }
                .onChange(of: text) { _, newText in
                    if focused, let parsed = Double(newText.replacingOccurrences(of: ",", with: ".")) {
                        value = parsed
                    }
                }
                .onChange(of: focused) { _, isFocused in
                    if !isFocused { text = String(format: format, value) }
                }
        }
    }
}

struct IntField: View {
    let label: String
    @Binding var value: Int

    var body: some View {
        NumberField(label: label, value: Binding(
            get: { Double(value) },
            set: { value = Int($0) }
        ), format: "%.0f")
    }
}

/// Card-style grouping used across all editor tabs.
/// Labeled free-text field (comma-separated lists, JSON snippets).
struct LabeledTextField: View {
    let label: String
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            if !label.isEmpty {
                Text(label).font(.caption).foregroundStyle(.secondary)
            }
            TextField(label, text: $text)
                .textFieldStyle(.roundedBorder)
                #if os(iOS)
                .autocorrectionDisabled()
                #endif
        }
    }
}

struct SectionCard<Content: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.headline)
            if let subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }
}

/// Compact three-component vector editor.
struct VectorEditor: View {
    let labels: [String]
    @Binding var values: [Double]

    init(labels: [String] = ["x", "y", "z"], values: Binding<[Double]>) {
        self.labels = labels
        self._values = values
    }

    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<3, id: \.self) { i in
                NumberField(label: labels.count > i ? labels[i] : "", value: Binding(
                    get: { values.count > i ? values[i] : 0 },
                    set: { newVal in
                        var v = values
                        while v.count < 3 { v.append(0) }
                        v[i] = newVal
                        values = v
                    }
                ))
            }
        }
    }
}

// MARK: - Notification banner

struct NotificationBanner: View {
    let notification: Notification

    var body: some View {
        Label(notification.message, systemImage: icon)
            .font(.callout.weight(.medium))
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(color.opacity(0.14), in: Capsule())
            .overlay(Capsule().strokeBorder(color.opacity(0.4)))
            .foregroundStyle(color)
    }

    private var color: Color {
        switch notification.kind {
        case .success: return .green
        case .error: return .red
        case .info: return .blue
        }
    }

    private var icon: String {
        switch notification.kind {
        case .success: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        case .info: return "info.circle.fill"
        }
    }
}

// MARK: - Connection status dot

struct ConnectionStatusView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(model.serverReachable ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(model.serverReachable ? "Backend connected" : "Backend offline")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
